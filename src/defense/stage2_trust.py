"""Stage 2: Trust scoring and re-ranking.

T(dᵢ) = 0.35·aᵢ + 0.35·cᵢ + 0.30·hᵢ

where:
  aᵢ = semantic alignment with other retrieved documents
  cᵢ = classifier confidence that the document is clean, computed as 1 - P(poisoned)
  hᵢ = query-document coherence, using retrieval score as a proxy

This stage takes documents that survived Stage 1, assigns each one a continuous
trust score, and re-ranks the documents before Stage 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config import (
    TOP_K_AFTER_TRUST,
    TRUST_WEIGHT_ALIGNMENT,
    TRUST_WEIGHT_CLASSIFIER,
    TRUST_WEIGHT_COHERENCE,
)


def normalize_scores(scores: list[float] | np.ndarray) -> list[float]:
    """Normalize a list of scores to [0, 1]."""
    scores_array = np.array(scores, dtype=float)

    if len(scores_array) == 0:
        return []

    score_min = scores_array.min()
    score_max = scores_array.max()

    if score_max - score_min < 1e-10:
        return [1.0 for _ in scores_array]

    normalized = (scores_array - score_min) / (score_max - score_min)
    return normalized.tolist()


def compute_semantic_alignment(doc_embeddings: np.ndarray) -> list[float]:
    """
    For each document, compute average cosine similarity to all other documents.

    The MiniLM retriever embeddings are assumed to be L2-normalized, so dot
    product is equivalent to cosine similarity.
    """
    doc_embeddings = np.asarray(doc_embeddings, dtype=float)

    if len(doc_embeddings) == 0:
        return []

    if len(doc_embeddings) == 1:
        return [1.0]

    sim_matrix = doc_embeddings @ doc_embeddings.T

    alignments = []

    for i in range(len(doc_embeddings)):
        other_similarities = [
            sim_matrix[i, j]
            for j in range(len(doc_embeddings))
            if j != i
        ]

        avg_alignment = float(np.mean(other_similarities))
        alignments.append(avg_alignment)

    return normalize_scores(alignments)


def compute_trust_scores(
    doc_embeddings: np.ndarray,
    classifier_scores: list[float],
    retrieval_scores: list[float],
    weights: tuple[float, float, float] = (
        TRUST_WEIGHT_ALIGNMENT,
        TRUST_WEIGHT_CLASSIFIER,
        TRUST_WEIGHT_COHERENCE,
    ),
) -> list[float]:
    """
    Compute trust scores for all documents.

    Args:
        doc_embeddings:
            Embeddings for the documents that survived Stage 1.

        classifier_scores:
            Stage 1 scores, where each value is P(poisoned).

        retrieval_scores:
            Retriever similarity scores. These are used as the query-document
            coherence proxy.

        weights:
            Weights for semantic alignment, classifier confidence, and coherence.

    Returns:
        A list of trust scores, one per document.
    """
    num_docs = len(doc_embeddings)

    if num_docs == 0:
        return []

    if not (
        len(classifier_scores) == num_docs
        and len(retrieval_scores) == num_docs
    ):
        raise ValueError(
            "doc_embeddings, classifier_scores, and retrieval_scores must have the same length"
        )

    wa, wc, wh = weights

    if abs((wa + wc + wh) - 1.0) > 1e-6:
        raise ValueError("Trust scoring weights must sum to 1.0")

    alignment_scores = compute_semantic_alignment(doc_embeddings)

    # Stage 1 outputs P(poisoned), so clean confidence is 1 - P(poisoned).
    clean_confidence_scores = [1.0 - score for score in classifier_scores]

    # Retrieval score is used as query-document coherence proxy.
    coherence_scores = normalize_scores(retrieval_scores)

    trust_scores = []

    for alignment, clean_confidence, coherence in zip(
        alignment_scores,
        clean_confidence_scores,
        coherence_scores,
    ):
        trust_score = (
            wa * alignment
            + wc * clean_confidence
            + wh * coherence
        )

        trust_scores.append(float(trust_score))

    return trust_scores


def rerank_and_filter(
    documents: list[str],
    trust_scores: list[float],
    top_k: int = TOP_K_AFTER_TRUST,
) -> list[tuple[str, float]]:
    """
    Sort documents by trust score descending and return top-k document-score pairs.
    """
    if len(documents) != len(trust_scores):
        raise ValueError("documents and trust_scores must have the same length")

    ranked_documents = sorted(
        zip(documents, trust_scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked_documents[:top_k]


def score_and_rerank_documents(
    documents: list[str],
    doc_embeddings: np.ndarray,
    classifier_scores: list[float],
    retrieval_scores: list[float],
    top_k: int = TOP_K_AFTER_TRUST,
) -> list[dict[str, Any]]:
    """
    Full Stage 2 helper function.

    This is the easiest function to call from the main pipeline.

    Returns:
        A list of dictionaries containing document text, trust score, and rank.
    """
    trust_scores = compute_trust_scores(
        doc_embeddings=doc_embeddings,
        classifier_scores=classifier_scores,
        retrieval_scores=retrieval_scores,
    )

    ranked_pairs = rerank_and_filter(
        documents=documents,
        trust_scores=trust_scores,
        top_k=top_k,
    )

    output = []

    for rank, (document, trust_score) in enumerate(ranked_pairs, start=1):
        output.append(
            {
                "rank": rank,
                "document": document,
                "trust_score": trust_score,
            }
        )

    return output
