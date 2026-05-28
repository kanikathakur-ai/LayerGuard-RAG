"""Stage 3: NLI contradiction graph.

Uses DeBERTa-v3-large-mnli-fever-anli (no fine-tuning) to detect
contradictions between retrieved documents. Flags docs that contradict ≥2
others as outliers.
"""

from itertools import combinations

import numpy as np
import torch
from transformers import pipeline as hf_pipeline

from config import CONTRADICTION_THRESHOLD, MIN_CONTRADICTIONS_TO_FLAG, NLI_MODEL


def load_nli_model(model_name: str = NLI_MODEL):
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline(
        "text-classification", model=model_name, device=device, top_k=None
    )


def compute_pairwise_contradictions(
    documents: list[str],
    nli_pipeline,
    max_length: int = 512,
) -> np.ndarray:
    """Compute an n×n contradiction probability matrix.

    Entry [i,j] = P(doc_i contradicts doc_j).
    """
    n = len(documents)
    matrix = np.zeros((n, n), dtype=float)

    pairs = list(combinations(range(n), 2))
    if not pairs:
        return matrix

    # Truncate documents to stay within 512-token limit
    truncated = [
        doc[:1000] for doc in documents
    ]  # rough char truncation; tokenizer handles the rest

    inputs = [{"text": truncated[i], "text_pair": truncated[j]} for i, j in pairs]
    results = nli_pipeline(inputs, truncation=True, max_length=max_length)

    for (i, j), label_scores in zip(pairs, results):
        contradiction_prob = next(
            (s["score"] for s in label_scores if s["label"].lower() == "contradiction"),
            0.0,
        )
        matrix[i, j] = contradiction_prob
        matrix[j, i] = contradiction_prob  # symmetric

    return matrix


def build_contradiction_graph(
    contradiction_matrix: np.ndarray,
    threshold: float = CONTRADICTION_THRESHOLD,
) -> dict[int, list[int]]:
    """Return adjacency dict: doc_idx → [list of docs it contradicts above threshold]."""
    n = contradiction_matrix.shape[0]
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and contradiction_matrix[i, j] > threshold:
                graph[i].append(j)
    return graph


def flag_outliers(
    graph: dict[int, list[int]],
    min_contradictions: int = MIN_CONTRADICTIONS_TO_FLAG,
) -> list[int]:
    """Return indices of docs that contradict ≥ min_contradictions other docs."""
    return [
        idx for idx, neighbors in graph.items() if len(neighbors) >= min_contradictions
    ]


def filter_outliers(
    documents: list[str],
    trust_scores: list[float],
    outlier_indices: list[int],
) -> list[tuple[str, float]]:
    """Remove outlier docs; return remaining (doc, trust_score) pairs."""
    return [
        (doc, score)
        for i, (doc, score) in enumerate(zip(documents, trust_scores))
        if i not in set(outlier_indices)
    ]
