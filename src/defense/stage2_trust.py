"""Stage 2: Trust scoring and re-ranking.

T(dᵢ) = 0.35·aᵢ + 0.35·cᵢ + 0.30·hᵢ
  aᵢ = semantic alignment (avg cosine sim to peers)
  cᵢ = classifier confidence (1 - P(poisoned))
  hᵢ = query-doc coherence (proxy: retrieval score, normalized)
"""

import numpy as np
from config import TRUST_WEIGHT_ALIGNMENT, TRUST_WEIGHT_CLASSIFIER, TRUST_WEIGHT_COHERENCE, TOP_K_AFTER_TRUST


def compute_semantic_alignment(doc_embeddings: np.ndarray) -> list[float]:
    """For each doc, compute avg cosine similarity to all other docs."""
    # Embeddings are assumed to be L2-normalized (from MiniLM retriever)
    sim_matrix = doc_embeddings @ doc_embeddings.T  # cosine sim since normalized
    n = len(doc_embeddings)
    alignments = []
    for i in range(n):
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        alignments.append(float(np.mean(others)) if others else 0.0)
    return alignments


def compute_trust_scores(
    doc_embeddings: np.ndarray,
    classifier_scores: list[float],  # P(poisoned) from Stage 1
    retrieval_scores: list[float],   # cosine sim from FAISS, used as coherence proxy
    weights: tuple[float, float, float] = (
        TRUST_WEIGHT_ALIGNMENT,
        TRUST_WEIGHT_CLASSIFIER,
        TRUST_WEIGHT_COHERENCE,
    ),
) -> list[float]:
    wa, wc, wh = weights

    alignments = compute_semantic_alignment(doc_embeddings)
    confidence = [1.0 - s for s in classifier_scores]

    # Normalize retrieval scores to [0, 1] as coherence proxy
    r = np.array(retrieval_scores, dtype=float)
    r_min, r_max = r.min(), r.max()
    coherence = ((r - r_min) / (r_max - r_min + 1e-10)).tolist()

    trust = [
        wa * a + wc * c + wh * h
        for a, c, h in zip(alignments, confidence, coherence)
    ]
    return trust


def rerank_and_filter(
    documents: list[str],
    trust_scores: list[float],
    top_k: int = TOP_K_AFTER_TRUST,
) -> list[tuple[str, float]]:
    """Sort by trust score descending, return top-k (doc, score) pairs."""
    ranked = sorted(zip(documents, trust_scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
