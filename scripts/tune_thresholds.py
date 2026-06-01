"""Grid search for τ₁ (Stage 1), τ₃ (Stage 3), and trust weights.

Usage:
    python scripts/tune_thresholds.py \
        --val-questions data/nq/val_questions.jsonl \
        --stage1-model results/stage1_classifier
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from itertools import product

import numpy as np
from sklearn.metrics import fbeta_score


def tune_stage1_threshold(val_scores: list[float], val_labels: list[int]) -> float:
    """Find τ₁ maximizing F2 on the validation set."""
    best_tau, best_f2 = 0.5, 0.0
    for tau in np.arange(0.1, 0.95, 0.05):
        preds = [1 if s > tau else 0 for s in val_scores]
        f2 = fbeta_score(val_labels, preds, beta=2, average="binary", zero_division=0)
        if f2 > best_f2:
            best_f2, best_tau = f2, tau
    print(f"Best τ₁ = {best_tau:.2f} (F2 = {best_f2:.4f})")
    return best_tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-questions", required=True)
    parser.add_argument("--stage1-model", required=True)
    args = parser.parse_args()
    print("Threshold tuning — implement after Stage 1 training is complete.")
    print("Use tune_stage1_threshold() with validation set scores and labels.")


if __name__ == "__main__":
    main()
