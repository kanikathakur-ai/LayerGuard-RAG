"""Defended RAG pipeline: retrieval → S1 → S2 → S3 → generator.

Self-contained orchestrator for the Stage 3 evaluation. Reuses the upstream
stages from ``src/`` and threads the corpus doc-id through every stage so that
"Recall@5 after defense" can be measured on the *surviving* set (the existing
``src/defense/pipeline.py`` drops ids after Stage 2).

The ``defense`` switch drives the ablation:
    none    : top-k retrieved, no filtering (vanilla)
    stage1  : + Stage 1 classifier filter
    stage12 : + Stage 2 trust re-rank
    full    : + Stage 3 NLI consistency check

Every config truncates the generator context to ``top_k_trust`` (default 5) docs
so configurations differ only in *which* docs survive, not how many.
"""

from __future__ import annotations

import time

import numpy as np

from config import (
    STAGE1_THRESHOLD,
    CONTRADICTION_THRESHOLD,
    TOP_K_RETRIEVAL,
    TOP_K_AFTER_TRUST,
    TRUST_WEIGHT_ALIGNMENT,
    TRUST_WEIGHT_CLASSIFIER,
    TRUST_WEIGHT_COHERENCE,
)
from src.retriever import retrieve
from src.generator import generate_answer
from src.defense.stage1_classifier import score_documents
from src.defense.stage2_trust import score_and_rerank_documents
from stage_3_jincheng.stage3_nli import apply_stage3, STAGE3_CLEAN_CONF_THRESHOLD

VALID_DEFENSES = ("none", "stage1", "stage12", "full")


def _poison_count(ids, poison_id_set) -> int:
    if not poison_id_set:
        return 0
    return sum(1 for i in ids if i in poison_id_set)


def run_defended(
    query: str,
    index,
    documents: list[str],
    encoder,
    doc_embeddings: np.ndarray,
    stage1_model=None,
    stage1_tokenizer=None,
    nli_pipeline=None,
    gen_model=None,
    gen_tokenizer=None,
    defense: str = "full",
    tau1: float = STAGE1_THRESHOLD,
    tau3: float = CONTRADICTION_THRESHOLD,
    clean_conf_threshold: float = STAGE3_CLEAN_CONF_THRESHOLD,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_trust: int = TOP_K_AFTER_TRUST,
    trust_weights: tuple = (
        TRUST_WEIGHT_ALIGNMENT,
        TRUST_WEIGHT_CLASSIFIER,
        TRUST_WEIGHT_COHERENCE,
    ),
    poison_id_set: set | None = None,
    generate: bool = True,
) -> dict:
    if defense not in VALID_DEFENSES:
        raise ValueError(f"defense must be one of {VALID_DEFENSES}, got {defense!r}")

    timings: dict[str, float] = {}
    diag: dict = {}

    # ---- Retrieval ----
    t = time.perf_counter()
    retrieved = retrieve(query, index, documents, encoder, k=top_k_retrieval)
    timings["retrieve_s"] = time.perf_counter() - t

    ret_docs = [d for d, _, _ in retrieved]
    ret_ids = [i for _, i, _ in retrieved]
    ret_scores = [s for _, _, s in retrieved]
    diag["n_retrieved"] = len(ret_ids)
    diag["retrieved_poison"] = _poison_count(ret_ids, poison_id_set)

    # ---- Config: none (vanilla) ----
    if defense == "none":
        final = list(zip(ret_docs, ret_ids))[:top_k_trust]
        final_docs = [d for d, _ in final]
        final_ids = [i for _, i in final]
        return _finish(
            query, final_docs, final_ids, diag, timings,
            gen_model, gen_tokenizer, generate, poison_id_set,
        )

    # ---- Stage 1: classifier filter ----
    t = time.perf_counter()
    poison_scores = score_documents(query, ret_docs, stage1_model, stage1_tokenizer)
    survivors = [
        (d, ps, rs, i)
        for d, ps, rs, i in zip(ret_docs, poison_scores, ret_scores, ret_ids)
        if ps <= tau1
    ]
    if not survivors:  # safety net: keep the least-suspicious doc
        m = min(range(len(poison_scores)), key=lambda k: poison_scores[k])
        survivors = [(ret_docs[m], poison_scores[m], ret_scores[m], ret_ids[m])]
    timings["stage1_s"] = time.perf_counter() - t
    diag["n_stage1_survivors"] = len(survivors)
    diag["stage1_survivor_poison"] = _poison_count([s[3] for s in survivors], poison_id_set)

    if defense == "stage1":
        chosen = survivors[:top_k_trust]  # survivors keep retrieval order
        final_docs = [s[0] for s in chosen]
        final_ids = [s[3] for s in chosen]
        return _finish(
            query, final_docs, final_ids, diag, timings,
            gen_model, gen_tokenizer, generate, poison_id_set,
        )

    # ---- Stage 2: trust scoring + re-rank ----
    s1_docs = [s[0] for s in survivors]
    s1_pscores = [s[1] for s in survivors]
    s1_rscores = [s[2] for s in survivors]
    s1_ids = [s[3] for s in survivors]
    s1_embs = np.array([doc_embeddings[i] for i in s1_ids])

    t = time.perf_counter()
    ranked = score_and_rerank_documents(
        documents=s1_docs,
        doc_embeddings=s1_embs,
        classifier_scores=s1_pscores,
        retrieval_scores=s1_rscores,
        top_k=top_k_trust,
        weights=trust_weights,
    )
    timings["stage2_s"] = time.perf_counter() - t

    # Re-associate ranked docs back to their corpus id + poison score (text match,
    # mirroring eval_stage2_metrics.py). Track used indices to handle duplicates.
    used = set()
    s2_docs, s2_ids, s2_trust, s2_clean_conf = [], [], [], []
    for item in ranked:
        for k, sd in enumerate(s1_docs):
            if k not in used and sd == item["document"]:
                used.add(k)
                s2_docs.append(sd)
                s2_ids.append(s1_ids[k])
                s2_trust.append(item["trust_score"])
                s2_clean_conf.append(1.0 - s1_pscores[k])
                break
    diag["n_stage2"] = len(s2_docs)
    diag["stage2_poison"] = _poison_count(s2_ids, poison_id_set)

    if defense == "stage12":
        return _finish(
            query, s2_docs, s2_ids, diag, timings,
            gen_model, gen_tokenizer, generate, poison_id_set,
        )

    # ---- Stage 3: NLI consistency check ----
    t = time.perf_counter()
    s3 = apply_stage3(
        documents=s2_docs,
        doc_ids=s2_ids,
        trust_scores=s2_trust,
        nli_pipeline=nli_pipeline,
        clean_confidences=s2_clean_conf,
        tau3=tau3,
        clean_conf_threshold=clean_conf_threshold,
    )
    timings["stage3_s"] = time.perf_counter() - t

    flagged_ids = [s2_ids[i] for i in s3["flagged"]]
    removed_ids = [s2_ids[i] for i in s3["removed"]]
    diag["stage3_flagged"] = len(s3["flagged"])
    diag["stage3_removed"] = len(s3["removed"])
    diag["stage3_poison_flagged"] = _poison_count(flagged_ids, poison_id_set)
    diag["stage3_poison_removed"] = _poison_count(removed_ids, poison_id_set)
    diag["stage3_clean_flagged"] = len(flagged_ids) - diag["stage3_poison_flagged"]
    diag["stage3_clean_removed"] = len(removed_ids) - diag["stage3_poison_removed"]
    diag["stage3_majority"] = s3["majority"]

    final_docs = [d for d, _, _ in s3["survivors"]]
    final_ids = s3["surviving_ids"]
    return _finish(
        query, final_docs, final_ids, diag, timings,
        gen_model, gen_tokenizer, generate, poison_id_set,
    )


def _finish(query, final_docs, final_ids, diag, timings,
            gen_model, gen_tokenizer, generate, poison_id_set):
    diag["n_final"] = len(final_ids)
    diag["final_poison"] = _poison_count(final_ids, poison_id_set)

    answer = None
    if generate and gen_model is not None:
        t = time.perf_counter()
        answer = generate_answer(query, final_docs, gen_model, gen_tokenizer)
        timings["generate_s"] = time.perf_counter() - t
    timings["total_s"] = sum(timings.values())

    return {
        "answer": answer,
        "final_docs": final_docs,
        "surviving_ids": final_ids,
        "diagnostics": diag,
        "timings": timings,
    }
