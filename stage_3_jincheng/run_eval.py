"""End-to-end evaluation of Stage 3 (LayerGuard-RAG), NQ only.

Runs the ablation  none → stage1 → stage12 → full  over a subsample of NQ, plus
a clean (no-poison) baseline, and reports:

  - ASR (attacker target-answer substring match, on poisoned target queries)
  - Recall@5 of the gold doc after defense (on queries with a known gold_doc_id)
  - F1 / EM on clean queries (answer quality not degraded by the defense)
  - per-stage latency
  - intrinsic Stage-3 metrics (poison reaching S3 / flagged / removed, gold kept)

Poison "dose" axis (instead of corpus ratio):
    The PoisonedRAG-style attack here generates a fixed, small number of poison
    docs (n_target * dose). ``inject_into_corpus`` caps injection at that count,
    so a corpus poison-ratio of 5/10/20% collapses to the same thing. What
    actually determines whether poison reaches Stage 3 is how many poison docs
    land in each query's top-k — i.e. the per-question DOSE. So we ablate over
    dose ∈ {1,3,5}: dose=1 → poison is a minority in the retrieved set,
    dose=5 → poison can be the majority. Poison is appended to the corpus
    (positions are irrelevant to content-based FAISS retrieval) so original
    gold_doc_id values stay valid.

Usage (from repo root):
    uv run python stage_3_jincheng/run_eval.py                 # full run
    uv run python stage_3_jincheng/run_eval.py --quick         # tiny smoke
    uv run python stage_3_jincheng/run_eval.py --no-generate   # intrinsic only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, NLI_MODEL
from src.retriever import load_documents, load_index, inject_documents
from src.defense.stage1_classifier import load_classifier
from src.attacks.inject_poison import run_attack
from src.attacks.poisonedrag_attack import (
    load_poisonedrag,
    poisonedrag_questions,
    resolve_gold_doc_ids,
)
from eval.metrics import compute_asr, compute_f1, compute_em

DOCS_PATH = os.path.join(ROOT, "data/nq/documents.jsonl")
QUESTIONS_PATH = os.path.join(ROOT, "data/nq/test_questions.jsonl")
INDEX_PATH = os.path.join(ROOT, "data/nq/faiss.index")
EMB_PATH = os.path.join(ROOT, "data/nq/faiss_embeddings.npy")
STAGE1_MODEL_PATH = "michchicken/layerguard-stage1"
OUTPUT_PATH = os.path.join(ROOT, "results/stage3_eval.json")

GENERATOR_CANDIDATES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",  # ungated fallback
]


def load_questions(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_generator_with_fallback(forced=None):
    """Try generators in order until one loads; return (model, tok, name)."""
    from src.generator import load_generator

    candidates = [forced] if forced else GENERATOR_CANDIDATES
    last = None
    for name in candidates:
        try:
            print(f"[gen] loading {name} ...")
            model, tok = load_generator(name)
            print(f"[gen] using {name}")
            return model, tok, name
        except Exception as e:  # gated / missing / OOM → try next
            print(f"[gen] {name} failed: {type(e).__name__}: {e}")
            last = e
    raise RuntimeError(f"No generator could be loaded; last error: {last}")


def build_poisoned_corpus(target_qs, dose, clean_docs, clean_emb, encoder,
                          poison_examples=None):
    """Append `dose` poison docs per target question; return contaminated state.

    poison_examples: pre-built list[PoisonedExample] (used for --attack poisonedrag).
    When None, generates template-based poison from target_qs (original behaviour).
    dose caps the number of docs taken per example (max 5 for PoisonedRAG).
    """
    if poison_examples is None:
        attack_targets = [
            {"question": q["question"], "target_answer": q["target_answer"]}
            for q in target_qs
        ]
        examples = run_attack(attack_targets, n_docs_per_question=dose, seed=42)
    else:
        examples = [
            type(ex)(
                question=ex.question,
                target_answer=ex.target_answer,
                poisoned_docs=ex.poisoned_docs[:dose],
            )
            for ex in poison_examples
        ]
    poison_docs = [d for ex in examples for d in ex.poisoned_docs]

    index, _ = load_index(INDEX_PATH, EMB_PATH)  # fresh clean index each call
    index, emb, docs = inject_documents(
        index, clean_emb, list(clean_docs), poison_docs, encoder
    )
    poison_id_set = set(range(len(clean_docs), len(docs)))
    return index, emb, docs, poison_id_set


def evaluate(condition, defense, index, emb, docs, poison_id_set,
             target_qs, clean_qs, models, generate):
    """Run one (condition, defense) cell; return a metrics dict."""
    from stage_3_jincheng.defended_pipeline import run_defended

    encoder, s1_model, s1_tok, nli, gen_model, gen_tok = models
    eval_qs = [("target", q) for q in target_qs] + [("clean", q) for q in clean_qs]

    target_preds, target_golds = [], []   # ASR
    clean_preds, clean_golds = [], []      # F1/EM
    recall_hits = recall_total = 0
    stage_times: dict[str, list] = {}
    intr = {k: 0 for k in (
        "poison_into_s3", "stage3_flagged", "stage3_removed",
        "stage3_poison_removed", "stage3_clean_removed", "n_with_s3",
        "retrieved_poison", "stage1_survivor_poison", "final_poison",
        "gold_in_final_target",
    )}
    n_target = 0

    for kind, q in eval_qs:
        out = run_defended(
            query=q["question"], index=index, documents=docs, encoder=encoder,
            doc_embeddings=emb, stage1_model=s1_model, stage1_tokenizer=s1_tok,
            nli_pipeline=nli, gen_model=gen_model, gen_tokenizer=gen_tok,
            defense=defense, poison_id_set=poison_id_set, generate=generate,
        )
        d = out["diagnostics"]
        sids = out["surviving_ids"][:5]

        # Recall@5 on the surviving (post-defense) set
        gid = q.get("gold_doc_id")
        if gid is not None:
            recall_total += 1
            hit = gid in sids
            recall_hits += int(hit)
            if kind == "target":
                intr["gold_in_final_target"] += int(hit)

        # ASR / QA
        if kind == "target":
            n_target += 1
            if out["answer"] is not None:
                target_preds.append(out["answer"])
                target_golds.append(q["target_answer"])
            intr["retrieved_poison"] += d.get("retrieved_poison", 0)
            intr["stage1_survivor_poison"] += d.get("stage1_survivor_poison", 0)
            intr["final_poison"] += d.get("final_poison", 0)
            if "stage2_poison" in d:
                intr["poison_into_s3"] += d.get("stage2_poison", 0)
            if "stage3_flagged" in d:
                intr["n_with_s3"] += 1
                intr["stage3_flagged"] += d["stage3_flagged"]
                intr["stage3_removed"] += d["stage3_removed"]
                intr["stage3_poison_removed"] += d.get("stage3_poison_removed", 0)
                intr["stage3_clean_removed"] += d.get("stage3_clean_removed", 0)
        else:
            if out["answer"] is not None:
                clean_preds.append(out["answer"])
                clean_golds.append(q["answer"])

        for k, v in out["timings"].items():
            stage_times.setdefault(k, []).append(v)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "condition": condition,
        "defense": defense,
        "n_target": n_target,
        "n_clean": len(clean_qs),
        "asr": compute_asr(target_preds, target_golds) if target_preds else None,
        "recall_at_5": (recall_hits / recall_total) if recall_total else None,
        "recall_n": recall_total,
        "f1": compute_f1(clean_preds, clean_golds) if clean_preds else None,
        "em": compute_em(clean_preds, clean_golds) if clean_preds else None,
        "latency_ms": {k: round(1000 * mean(v), 2) for k, v in stage_times.items()},
        "intrinsic": intr,
    }
    # Stage-3 detection precision/recall on poison (over target queries)
    if intr["stage3_removed"]:
        metrics["stage3_precision_poison"] = round(
            intr["stage3_poison_removed"] / intr["stage3_removed"], 4)
    if intr["poison_into_s3"]:
        metrics["stage3_recall_poison"] = round(
            intr["stage3_poison_removed"] / intr["poison_into_s3"], 4)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=100)
    ap.add_argument("--n-clean", type=int, default=50)
    ap.add_argument("--doses", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--configs", nargs="+",
                    default=["none", "stage1", "stage12", "full"])
    ap.add_argument("--nli-model", default=NLI_MODEL)
    ap.add_argument("--generator-model", default=None)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--output", default=OUTPUT_PATH)
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run: 3 targets, 2 clean, dose=[5]")
    ap.add_argument("--attack", choices=["template", "poisonedrag"], default="template",
                    help="Attack source: template (original) or poisonedrag (GPT-4 generated)")
    ap.add_argument("--adv-path", default="PoisonedRAG/results/adv_targeted_results/nq.json",
                    help="Path to adv_targeted_results JSON (used when --attack poisonedrag)")
    ap.add_argument("--resolve-gold", action="store_true",
                    help="Attempt to fill gold_doc_id for PoisonedRAG targets via retrieval")
    args = ap.parse_args()

    if args.quick:
        args.n_target, args.n_clean, args.doses = 3, 2, [5]

    generate = not args.no_generate
    print(f"Config: n_target={args.n_target} n_clean={args.n_clean} "
          f"doses={args.doses} configs={args.configs} generate={generate} "
          f"attack={args.attack}")

    print("Loading corpus + index + embeddings + encoder ...")
    clean_docs = load_documents(DOCS_PATH)
    clean_emb = np.load(EMB_PATH)
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    questions = load_questions(QUESTIONS_PATH)

    # --- Target question selection ---
    poison_examples = None
    if args.attack == "poisonedrag":
        print(f"Loading PoisonedRAG adversarial data from {args.adv_path} ...")
        poison_examples = load_poisonedrag(args.adv_path, n_docs=5)
        target_qs = poisonedrag_questions(args.adv_path)[: args.n_target]
        if args.resolve_gold:
            print("Resolving gold_doc_ids for PoisonedRAG targets via retrieval ...")
            clean_index, _ = load_index(INDEX_PATH, EMB_PATH)
            resolve_gold_doc_ids(target_qs, clean_index, list(clean_docs), encoder)
            filled = sum(1 for q in target_qs if q["gold_doc_id"] is not None)
            print(f"  gold_doc_id filled for {filled}/{len(target_qs)} targets")
    else:
        target_qs = questions[: args.n_target]

    clean_qs = questions[200 : 200 + args.n_clean]  # never poisoned

    print("Loading Stage 1 classifier ...")
    s1_model, s1_tok = load_classifier(STAGE1_MODEL_PATH)

    nli = None
    if "full" in args.configs:
        from stage_3_jincheng.stage3_nli import load_nli
        print(f"Loading Stage 3 NLI model ({args.nli_model}) ...")
        nli = load_nli(args.nli_model)

    gen_model = gen_tok = gen_name = None
    if generate:
        gen_model, gen_tok, gen_name = load_generator_with_fallback(args.generator_model)

    models = (encoder, s1_model, s1_tok, nli, gen_model, gen_tok)

    clean_configs = [c for c in ("none", "full") if c in args.configs]
    cells_total = len(clean_configs) + len(args.doses) * len(args.configs)

    results = {
        "meta": {
            "n_target": args.n_target, "n_clean": args.n_clean,
            "doses": args.doses, "configs": args.configs,
            "nli_model": args.nli_model, "generator_model": gen_name,
            "generate": generate,
            "attack": args.attack,
            "adv_path": args.adv_path if args.attack == "poisonedrag" else None,
            "status": "running",
            "cells_total": cells_total,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "cells": [],
    }

    def _dump(results, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    t_start = time.perf_counter()

    # ---- Clean baseline (no poison) ----
    print("\n=== CLEAN baseline (no poison) ===")
    cindex, _ = load_index(INDEX_PATH, EMB_PATH)
    for defense in clean_configs:
        print(f"  [clean] defense={defense}")
        m = evaluate("clean", defense, cindex, clean_emb, clean_docs, set(),
                     target_qs, clean_qs, models, generate)
        m["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["cells"].append(m)
        _dump(results, args.output)

    # ---- Poisoned conditions, by dose ----
    for dose in args.doses:
        print(f"\n=== POISONED dose={dose} (poison docs / target question) ===")
        index, emb, docs, pset = build_poisoned_corpus(
            target_qs, dose, clean_docs, clean_emb, encoder,
            poison_examples=poison_examples)
        print(f"  injected {len(pset)} poison docs; corpus now {len(docs)} docs")
        for defense in args.configs:
            print(f"  [dose={dose}] defense={defense}")
            m = evaluate(f"poison_dose{dose}", defense, index, emb, docs, pset,
                         target_qs, clean_qs, models, generate)
            m["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            results["cells"].append(m)
            asr = m["asr"]
            print(f"    ASR={asr} Recall@5={m['recall_at_5']} "
                  f"F1={m['f1']} S3_removed={m['intrinsic']['stage3_removed']} "
                  f"S3_poison_removed={m['intrinsic']['stage3_poison_removed']}")
            _dump(results, args.output)

    results["meta"]["wall_clock_s"] = round(time.perf_counter() - t_start, 1)
    results["meta"]["status"] = "complete"
    _dump(results, args.output)
    print(f"\nSaved results to {args.output}  ({results['meta']['wall_clock_s']}s)")

    _print_summary(results)


def _print_summary(results):
    print("\n================ SUMMARY ================")
    hdr = f"{'condition':<14}{'defense':<9}{'ASR':>7}{'Recall@5':>10}{'F1':>7}{'EM':>7}"
    print(hdr)
    print("-" * len(hdr))
    for m in results["cells"]:
        def fmt(x, p=3):
            return f"{x:.{p}f}" if isinstance(x, (int, float)) else "  -  "
        print(f"{m['condition']:<14}{m['defense']:<9}{fmt(m['asr']):>7}"
              f"{fmt(m['recall_at_5']):>10}{fmt(m['f1']):>7}{fmt(m['em']):>7}")


if __name__ == "__main__":
    main()
