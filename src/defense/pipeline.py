"""Full 3-stage LayerGuard-RAG pipeline orchestrator."""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from config import (
    TOP_K_RETRIEVAL, TOP_K_AFTER_TRUST, STAGE1_THRESHOLD,
    CONTRADICTION_THRESHOLD, MIN_CONTRADICTIONS_TO_FLAG,
    TRUST_WEIGHT_ALIGNMENT, TRUST_WEIGHT_CLASSIFIER, TRUST_WEIGHT_COHERENCE,
)
from src.retriever import retrieve
from src.generator import generate_answer
from src.defense.stage1_classifier import score_documents
from src.defense.stage2_trust import compute_trust_scores, rerank_and_filter
from src.defense.stage3_nli import (
    compute_pairwise_contradictions, build_contradiction_graph,
    flag_outliers, filter_outliers,
)


def defend_and_answer(
    query: str,
    index,
    documents: list[str],
    encoder: SentenceTransformer,
    doc_embeddings: np.ndarray,
    model,
    tokenizer,
    stage1_model,
    stage1_tokenizer,
    nli_pipeline,
    tau1: float = STAGE1_THRESHOLD,
    tau3: float = CONTRADICTION_THRESHOLD,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_trust: int = TOP_K_AFTER_TRUST,
    min_contradictions: int = MIN_CONTRADICTIONS_TO_FLAG,
    trust_weights: tuple = (TRUST_WEIGHT_ALIGNMENT, TRUST_WEIGHT_CLASSIFIER, TRUST_WEIGHT_COHERENCE),
) -> dict:
    """End-to-end defended query answering.

    Returns a dict with answer, per-stage doc lists, scores, and timings.
    """
    timings = {}

    # Retrieval
    t0 = time.perf_counter()
    retrieved = retrieve(query, index, documents, encoder, k=top_k_retrieval)
    timings["retrieve_s"] = time.perf_counter() - t0

    ret_docs = [doc for doc, _, _ in retrieved]
    ret_ids = [doc_id for _, doc_id, _ in retrieved]
    ret_scores = [score for _, _, score in retrieved]

    # Stage 1: binary classifier filter
    t1 = time.perf_counter()
    poison_scores = score_documents(query, ret_docs, stage1_model, stage1_tokenizer)
    s1_survivors = [(doc, s, ret_scores[i], ret_ids[i])
                    for i, (doc, s) in enumerate(zip(ret_docs, poison_scores))
                    if s <= tau1]
    if not s1_survivors:
        s1_survivors = [(ret_docs[0], poison_scores[0], ret_scores[0], ret_ids[0])]
    s1_docs, s1_classifier_scores, s1_ret_scores, s1_ids = zip(*s1_survivors)
    s1_filtered = [doc for doc, s in zip(ret_docs, poison_scores) if s > tau1]
    timings["stage1_s"] = time.perf_counter() - t1

    # Stage 2: trust scoring + re-rank
    t2 = time.perf_counter()
    s1_embs = np.array([doc_embeddings[doc_id] for doc_id in s1_ids])
    trust = compute_trust_scores(s1_embs, list(s1_classifier_scores), list(s1_ret_scores), weights=trust_weights)
    s2_ranked = rerank_and_filter(list(s1_docs), trust, top_k=top_k_trust)
    s2_docs = [doc for doc, _ in s2_ranked]
    s2_trust_scores = [score for _, score in s2_ranked]
    timings["stage2_s"] = time.perf_counter() - t2

    # Stage 3: NLI contradiction check
    t3 = time.perf_counter()
    if len(s2_docs) > 1:
        contradiction_matrix = compute_pairwise_contradictions(s2_docs, nli_pipeline)
        graph = build_contradiction_graph(contradiction_matrix, threshold=tau3)
        outliers = flag_outliers(graph, min_contradictions=min_contradictions)
        s3_survivors = filter_outliers(s2_docs, s2_trust_scores, outliers)
        s3_flagged = [s2_docs[i] for i in outliers]
    else:
        s3_survivors = list(zip(s2_docs, s2_trust_scores))
        s3_flagged = []
    timings["stage3_s"] = time.perf_counter() - t3

    final_docs = [doc for doc, _ in s3_survivors] if s3_survivors else s2_docs[:1]

    # Generate answer
    tg = time.perf_counter()
    answer = generate_answer(query, final_docs, model, tokenizer)
    timings["generate_s"] = time.perf_counter() - tg
    timings["total_s"] = sum(timings.values())

    return {
        "answer": answer,
        "surviving_docs": final_docs,
        "stage1_filtered": s1_filtered,
        "stage3_flagged": s3_flagged,
        "trust_scores": s2_trust_scores,
        "doc_scores": list(zip(ret_ids, ret_scores)),
        "timings": timings,
    }
