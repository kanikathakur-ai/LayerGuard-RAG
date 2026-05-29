"""LayerGuard-RAG Stage 3 (fresh implementation by Jincheng).

Cross-document consistency checker: among the documents that survive Stages 1
and 2, use NLI to flag docs that contradict the consensus of their peers and
drop them (with a robust trust gate) before the LLM generator sees them.

This package is self-contained but *reuses* the upstream stages from ``src/``
rather than reimplementing them. Entry points:

- ``stage3_nli``        — the Stage 3 module (NLI contradiction graph + gate)
- ``defended_pipeline`` — retrieval → S1 → S2 → S3 → generator, ablatable
- ``run_eval``          — end-to-end ablation + intrinsic Stage-3 metrics
- ``isolated_stage3_test`` — generation-free characterization of Stage 3
"""
