"""Failure mode categorization for LayerGuard-RAG.

For each failure (poisoned doc survived all 3 stages), categorize whether
the miss was due to Stage 1, trust scoring, NLI, or genuine alignment.
"""

from dataclasses import dataclass
from enum import Enum


class FailureMode(str, Enum):
    STAGE1_MISS = "stage1_miss"           # Classifier didn't flag it
    TRUST_TOO_HIGH = "trust_too_high"     # Survived trust re-ranking
    NLI_MISS = "nli_miss"                 # NLI didn't detect contradiction
    GENUINE_ALIGNMENT = "genuine_alignment"  # Poison doc genuinely aligned with clean docs


@dataclass
class FailureCase:
    question: str
    poison_doc: str
    generated_answer: str
    target_answer: str
    stage1_score: float         # P(poisoned) from Stage 1
    trust_score: float
    contradiction_count: int    # how many docs this one contradicted
    mode: FailureMode


def categorize_failure(case: FailureCase, tau1: float = 0.5, tau3: float = 0.7, min_contradictions: int = 2) -> FailureMode:
    if case.stage1_score < tau1:
        return FailureMode.STAGE1_MISS
    if case.trust_score > 0.6:
        return FailureMode.GENUINE_ALIGNMENT
    if case.contradiction_count < min_contradictions:
        return FailureMode.NLI_MISS
    return FailureMode.TRUST_TOO_HIGH


def summarize_failures(cases: list[FailureCase]) -> dict:
    counts = {mode: 0 for mode in FailureMode}
    for c in cases:
        counts[c.mode] += 1
    total = len(cases)
    return {mode.value: {"count": cnt, "pct": cnt / total if total else 0} for mode, cnt in counts.items()}
