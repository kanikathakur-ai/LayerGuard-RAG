"""Stage 3: Cross-Document Consistency Checker (NLI contradiction graph).

Faithful to the LayerGuard-RAG proposal (Methods, Stage 3):

  - Model: DeBERTa-v3-large fine-tuned on MNLI+FEVER+ANLI+LiNG+WANLI
    (``config.NLI_MODEL``), used as-is (no fine-tuning).
  - For each pair (d_i, d_j) among the docs that survived Stages 1-2 compute the
    contradiction probability ``P(contradiction | d_i, d_j)``. NLI is directional,
    so we evaluate *both* orderings and take the max as the undirected edge weight.
  - Build a contradiction graph G: an (undirected) edge between i and j when the
    edge weight exceeds tau3 (``config.CONTRADICTION_THRESHOLD``).
  - Flag a document as an outlier if it contradicts >= ceil((n-1)/2) peers, where
    n is the *actual* number of surviving docs (computed dynamically — not the
    hard-coded constant, so it adapts when Stage 1 over-filters and n < 5).

Trust gate (decision: gate robustly inside Stage 3, leave Stage 2 untouched).
Stage 2's trust scores are min-max normalized *per query*, so the raw value is a
within-set relative score, not an absolute one — a fixed numeric cutoff on it is
meaningless. So a flagged outlier is removed only when two robust signals agree:

  1. it sits in the **bottom half by Stage-2 trust rank** within the surviving
     set (relative — robust to the normalization), AND
  2. its **Stage-1 clean-confidence** ``1 - P(poisoned)`` is below
     ``clean_conf_threshold`` (absolute — transfers across queries).

A flagged doc that is high-rank OR high-clean-confidence is kept (anti-over-filter).
When clean-confidences are unavailable (e.g. the isolated test that bypasses
Stage 1), condition (2) is treated as satisfied so the gate reduces to the
rank test.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from config import NLI_MODEL, CONTRADICTION_THRESHOLD
from stage_3_jincheng.device import get_device, hf_pipeline_device

# Absolute Stage-1 clean-confidence below which a flagged outlier may be removed.
# Stage-3-local (not in config.py) by design — Stage 3 owns its gate.
STAGE3_CLEAN_CONF_THRESHOLD = 0.5

# Rough character cap before the tokenizer's own 512-token truncation kicks in.
_MAX_CHARS = 1200


def load_nli(model_name: str = NLI_MODEL, device: str | None = None):
    """Load the NLI cross-encoder as a HuggingFace text-classification pipeline.

    Returns a callable pipeline that, given {"text", "text_pair"} inputs, yields
    per-label score dicts (top_k=None → all labels).
    """
    from transformers import pipeline as hf_pipeline

    dev = device or get_device()
    return hf_pipeline(
        "text-classification",
        model=model_name,
        device=hf_pipeline_device(dev),
        top_k=None,
    )


def _contradiction_score(label_scores) -> float:
    """Pull the 'contradiction' probability out of a pipeline result."""
    for s in label_scores:
        if "contradiction" in s["label"].lower():
            return float(s["score"])
    return 0.0


def pairwise_contradictions(
    documents: list[str],
    nli_pipeline,
    max_length: int = 512,
    batch_size: int = 16,
) -> np.ndarray:
    """Symmetric n×n matrix of contradiction probabilities.

    ``matrix[i, j]`` = max( P(contra | d_i, d_j), P(contra | d_j, d_i) ).
    Diagonal is 0. Both orderings are scored because NLI is directional.
    """
    n = len(documents)
    matrix = np.zeros((n, n), dtype=float)
    if n < 2:
        return matrix

    truncated = [d[:_MAX_CHARS] for d in documents]
    pairs = list(combinations(range(n), 2))

    # Two directed inputs per unordered pair, batched into one pipeline call.
    inputs = []
    for i, j in pairs:
        inputs.append({"text": truncated[i], "text_pair": truncated[j]})
        inputs.append({"text": truncated[j], "text_pair": truncated[i]})

    results = nli_pipeline(
        inputs, truncation=True, max_length=max_length, batch_size=batch_size
    )

    for k, (i, j) in enumerate(pairs):
        fwd = _contradiction_score(results[2 * k])
        bwd = _contradiction_score(results[2 * k + 1])
        w = max(fwd, bwd)
        matrix[i, j] = w
        matrix[j, i] = w
    return matrix


def build_graph(
    contradiction_matrix: np.ndarray,
    threshold: float = CONTRADICTION_THRESHOLD,
) -> dict[int, list[int]]:
    """Adjacency dict: i → [j with an undirected contradiction edge above threshold]."""
    n = contradiction_matrix.shape[0]
    graph: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and contradiction_matrix[i, j] > threshold:
                graph[i].append(j)
    return graph


def majority_threshold(n: int) -> int:
    """ceil((n-1)/2) — # of contradicting peers needed to flag as an outlier."""
    return math.ceil((n - 1) / 2) if n > 1 else 1


def flag_outliers(graph: dict[int, list[int]], n: int) -> list[int]:
    """Indices that contradict >= majority_threshold(n) peers."""
    m = majority_threshold(n)
    return [i for i, neigh in graph.items() if len(neigh) >= m]


def _bottom_half_by_trust(trust_scores: list[float]) -> set[int]:
    """Indices in the lower half when sorted by trust descending (ties → lower).

    For n docs the top ceil(n/2) ranks are 'kept'; the rest are 'bottom half'.
    """
    n = len(trust_scores)
    if n <= 1:
        return set()
    order = sorted(range(n), key=lambda i: trust_scores[i], reverse=True)
    cutoff = math.ceil(n / 2)  # positions [cutoff:] are bottom half
    return set(order[cutoff:])


def apply_stage3(
    documents: list[str],
    doc_ids: list[int],
    trust_scores: list[float],
    nli_pipeline,
    clean_confidences: list[float] | None = None,
    tau3: float = CONTRADICTION_THRESHOLD,
    clean_conf_threshold: float = STAGE3_CLEAN_CONF_THRESHOLD,
) -> dict:
    """Run Stage 3 over the surviving set.

    Args:
        documents:         surviving doc texts (Stage 2 output, ranked).
        doc_ids:           corpus doc-id for each surviving doc (parallel list).
        trust_scores:      Stage 2 trust score per surviving doc (parallel list).
        nli_pipeline:      output of ``load_nli``.
        clean_confidences: optional Stage-1 ``1 - P(poisoned)`` per doc; if None,
                           the gate's clean-confidence condition is skipped.
        tau3:              contradiction-edge threshold.
        clean_conf_threshold: absolute clean-confidence floor for removal.

    Returns dict with:
        survivors:        list of (doc_text, doc_id, trust_score) kept, in input order
        surviving_ids:    [doc_id, ...] kept (parallel to survivors)
        flagged:          indices (into input) flagged as contradiction outliers
        removed:          indices (into input) actually removed by the gate
        kept_despite_flag: flagged indices that the gate spared
        matrix:           the contradiction matrix (for analysis)
        majority:         majority_threshold(n) used
    """
    n = len(documents)
    base = {
        "matrix": np.zeros((n, n)),
        "flagged": [],
        "removed": [],
        "kept_despite_flag": [],
        "majority": majority_threshold(n),
    }
    if n <= 1:
        survivors = [
            (documents[i], doc_ids[i], trust_scores[i]) for i in range(n)
        ]
        base.update(survivors=survivors, surviving_ids=list(doc_ids))
        return base

    matrix = pairwise_contradictions(documents, nli_pipeline)
    graph = build_graph(matrix, threshold=tau3)
    flagged = flag_outliers(graph, n)

    bottom_half = _bottom_half_by_trust(trust_scores)
    removed: list[int] = []
    kept_despite_flag: list[int] = []
    for i in flagged:
        is_bottom = i in bottom_half
        low_conf = clean_confidences is None or clean_confidences[i] < clean_conf_threshold
        if is_bottom and low_conf:
            removed.append(i)
        else:
            kept_despite_flag.append(i)

    removed_set = set(removed)
    survivors = [
        (documents[i], doc_ids[i], trust_scores[i])
        for i in range(n)
        if i not in removed_set
    ]
    # Safety net: never return an empty context.
    if not survivors:
        best = max(range(n), key=lambda i: trust_scores[i])
        survivors = [(documents[best], doc_ids[best], trust_scores[best])]
        removed = [i for i in removed if i != best]

    return {
        "survivors": survivors,
        "surviving_ids": [s[1] for s in survivors],
        "flagged": flagged,
        "removed": removed,
        "kept_despite_flag": kept_despite_flag,
        "matrix": matrix,
        "majority": majority_threshold(n),
    }
