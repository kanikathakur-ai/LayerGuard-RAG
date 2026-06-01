"""Reproduce all LayerGuard-RAG experimental results.

Runs every experiment in the order they were originally produced:
  1. Fetch data (NQ corpus, FAISS index, embeddings) from HuggingFace
  2. Stage 2 trust-scoring metrics on NQ
  3. Stage 3 end-to-end ablation (dose x {none,stage1,stage12,full})
  4. PoisonedRAG GPT-4 adversarial attack evaluation (NQ, --resolve-gold: fills
     gold_doc_id via retrieval for ~91/100 targets so Recall@5 is meaningful)
  5. Template-generalization runs (5 templates, --quick mode)

Usage (from repo root):
    # Full run — takes ~1–2 hrs with a GPU:
    python scripts/reproduce_all.py

    # Fast smoke test — all steps, small sizes:
    python scripts/reproduce_all.py --quick

    # Run only specific steps (comma-separated):
    python scripts/reproduce_all.py --steps data,stage2,stage3

    # Skip data fetch if already downloaded:
    python scripts/reproduce_all.py --skip-data

    # Override generator (default: auto Llama-3.1-8B → Mistral-7B fallback):
    python scripts/reproduce_all.py --generator-model mistralai/Mistral-7B-Instruct-v0.3

All results are written to results/ (gitignored).
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATES = ["assertion", "encyclopedia", "narrative", "news", "textbook"]

ALL_STEPS = ["data", "stage2", "stage3", "poisonedrag", "templates"]


def run(cmd: list[str], label: str) -> None:
    """Print and execute a command; exit on failure."""
    print(f"\n{'='*70}")
    print(f"[{label}]")
    print("  " + " ".join(cmd))
    print("="*70)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nERROR: '{label}' exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)
    print(f"  [{label}] done in {elapsed:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce all LayerGuard-RAG experimental results."
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Pass --quick / smoke sizes to each driver for a fast sanity run.",
    )
    ap.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=(
            f"Comma-separated subset of steps to run "
            f"(default: all). Choices: {ALL_STEPS}"
        ),
    )
    ap.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip the data-fetch step (data/ already populated).",
    )
    ap.add_argument(
        "--generator-model",
        default=None,
        help=(
            "HuggingFace model ID for generation. "
            "Default (None) = auto-fallback: tries Llama-3.1-8B first, "
            "then Mistral-7B-Instruct-v0.3 (matches published results)."
        ),
    )
    ap.add_argument(
        "--train-stage1",
        action="store_true",
        help=(
            "Retrain the Stage 1 classifier before running evals. "
            "Default: use the published HF model michchicken/layerguard-stage1."
        ),
    )
    args = ap.parse_args()

    steps = {s.strip() for s in args.steps.split(",")}
    invalid = steps - set(ALL_STEPS)
    if invalid:
        ap.error(f"Unknown steps: {invalid}. Valid: {ALL_STEPS}")

    python = sys.executable  # same venv/interpreter as the caller

    print(f"\nLayerGuard-RAG reproduce_all.py")
    print(f"  steps      : {sorted(steps)}")
    print(f"  quick      : {args.quick}")
    print(f"  generator  : {args.generator_model or '(auto-fallback)'}")
    print(f"  train_stage1: {args.train_stage1}")

    # ------------------------------------------------------------------
    # Step 0: fetch data
    # ------------------------------------------------------------------
    if "data" in steps and not args.skip_data:
        run([python, "scripts/fetch_data.py"], label="data: fetch from HuggingFace")
    else:
        print("\n[data] skipped")

    # ------------------------------------------------------------------
    # Optional: retrain Stage 1
    # ------------------------------------------------------------------
    if args.train_stage1:
        run(
            [
                python, "scripts/train_stage1.py",
                "--train-data", "data/synthetic_train/train.jsonl",
                "--val-data",   "data/synthetic_train/val.jsonl",
                "--output-dir", "results/stage1_classifier",
            ],
            label="stage1: retrain classifier",
        )

    # ------------------------------------------------------------------
    # Step 1: Stage 2 metrics
    # ------------------------------------------------------------------
    if "stage2" in steps:
        run([python, "eval_stage2_metrics.py"], label="stage2: trust-scoring metrics on NQ")
    else:
        print("\n[stage2] skipped")

    # ------------------------------------------------------------------
    # Helper: shared flags for run_eval.py calls
    # ------------------------------------------------------------------
    def run_eval(extra_args: list[str], label: str) -> None:
        cmd = [python, "stage_3_jincheng/run_eval.py"] + extra_args
        if args.generator_model:
            cmd += ["--generator-model", args.generator_model]
        run(cmd, label=label)

    # ------------------------------------------------------------------
    # Step 2: Stage 3 end-to-end ablation (main results)
    # ------------------------------------------------------------------
    if "stage3" in steps:
        stage3_args = [
            "--output", "results/stage3_eval_full.json",
            "--doses", "1", "3", "5",
            "--configs", "none", "stage1", "stage12", "full",
        ]
        if args.quick:
            stage3_args += ["--quick"]
        run_eval(stage3_args, label="stage3: end-to-end ablation (NQ, dose∈{1,3,5})")
    else:
        print("\n[stage3] skipped")

    # ------------------------------------------------------------------
    # Step 3: PoisonedRAG adversarial attack
    # ------------------------------------------------------------------
    if "poisonedrag" in steps:
        adv_path = "PoisonedRAG/results/adv_targeted_results/nq.json"
        if not os.path.exists(os.path.join(ROOT, adv_path)):
            print(
                f"\n[poisonedrag] WARNING: {adv_path} not found — skipping. "
                "Run PoisonedRAG attack generation first or check the path."
            )
        else:
            pr_args = [
                "--attack", "poisonedrag",
                "--adv-path", adv_path,
                "--resolve-gold",
                "--output", "results/stage3_eval_poisonedrag.json",
            ]
            if args.quick:
                pr_args += ["--quick"]
            run_eval(pr_args, label="poisonedrag: adversarial attack eval (NQ)")
    else:
        print("\n[poisonedrag] skipped")

    # ------------------------------------------------------------------
    # Step 4: Template generalization (5 templates, always --quick)
    # ------------------------------------------------------------------
    if "templates" in steps:
        for tmpl in TEMPLATES:
            out = f"results/stage3_eval_template_{tmpl}_quick.json"
            run_eval(
                [
                    "--attack", "template",
                    "--template", tmpl,
                    "--quick",
                    "--output", out,
                ],
                label=f"templates: generalization — {tmpl}",
            )
    else:
        print("\n[templates] skipped")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("All requested steps completed. Results written to results/")
    print("  results/stage2_metrics.json")
    print("  results/stage3_eval_full.json")
    print("  results/stage3_eval_poisonedrag.json")
    for tmpl in TEMPLATES:
        print(f"  results/stage3_eval_template_{tmpl}_quick.json")
    print("="*70)


if __name__ == "__main__":
    main()
