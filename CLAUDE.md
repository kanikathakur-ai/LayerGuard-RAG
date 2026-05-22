# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LayerGuard-RAG** is a three-stage defense pipeline that protects RAG systems against data poisoning (PoisonedRAG-style) attacks. It sits between the retriever and the LLM generator and filters out poisoned documents before the LLM sees them. The project extends RAGuard (Kolhe et al., 2025), replacing their expensive ZKIP mechanism (k+1 LLM forward passes per query) with three cheaper, complementary stages.

## Environment Setup

Python 3.13 (see `.python-version`), managed with `uv`. Dependencies are declared in `pyproject.toml` / `uv.lock`:

```bash
uv sync                 # install all deps
uv run pre-commit install
```

For GPU with VRAM ≥16GB, swap `faiss-cpu` for `faiss-gpu` in `pyproject.toml`. Llama-3.1-8B is gated on HuggingFace — request access ahead of time. If blocked, `mistralai/Mistral-7B-Instruct-v0.3` is the configured fallback (see `config.GENERATOR_FALLBACK_MODEL`).

All scripts must be run from the **repo root** — they do `sys.path.insert(0, root)` and resolve paths relative to `config.py`.

Lint/format is wired through pre-commit (`black --line-length=88`, `isort --profile=black`, plus standard hygiene hooks). Run manually with:

```bash
uv run pre-commit run --all-files
```

## Key Commands

```bash
# Build FAISS index (one-time)
python scripts/build_index.py --docs data/nq/documents.jsonl

# Prepare NQ corpus / test questions
python scripts/prepare_nq.py

# Generate Stage 1 synthetic training data
python scripts/gen_stage1_data.py

# Train Stage 1 classifier
python scripts/train_stage1.py \
    --train-data data/synthetic_train/train.jsonl \
    --val-data data/synthetic_train/val.jsonl \
    --output-dir results/stage1_classifier

# Tune thresholds (τ₁, τ₃, trust weights) on val set
python scripts/tune_thresholds.py

# Full pipeline experiment (poisoned)
python scripts/run_experiments.py \
    --docs data/nq/documents.jsonl \
    --questions data/nq/test_questions.jsonl \
    --poison-ratio 0.05 \
    --output results/vanilla_rag_5pct.json

# Clean baseline
python scripts/run_experiments.py ... --no-poison
```

The trained Stage 1 model is published at `michchicken/layerguard-stage1` on HuggingFace; `data/` and `results/` are gitignored due to size (see `stage2_handoff.md` for where to fetch / regenerate them).

## Architecture

```
User Query → Retriever (top-10, FAISS + MiniLM) →
  Stage 1: DeBERTa-v3-base cross-encoder — drop if P(poisoned) > τ₁ →
  Stage 2: Trust scoring — re-rank by T(dᵢ) = 0.35·aᵢ + 0.35·cᵢ + 0.30·hᵢ, keep top-5 →
  Stage 3: NLI contradiction graph (DeBERTa-v3-large) — drop docs contradicting ≥2 peers →
  Llama-3.1-8B-Instruct (4-bit) → Answer
```

**Stage 2 trust components:** `aᵢ` = avg cosine sim to peer docs; `cᵢ` = 1 − P(poisoned) from Stage 1; `hᵢ` = query-doc coherence (inverse suspicion).

**Layout:**
- `src/retriever.py`, `src/generator.py`, `src/vanilla_rag.py` — retrieval, LLM wrapper, undefended baseline
- `src/defense/stage1_classifier.py`, `stage2_trust.py`, `stage3_nli.py` — the three filters
- `src/defense/pipeline.py:defend_and_answer()` — orchestrator; returns `{"answer", "surviving_docs", "stage1_filtered", "stage3_flagged", "trust_scores", "timings"}`
- `src/attacks/` — PoisonedRAG injection + synthetic training-data generation
- `src/baselines/` — perplexity filter and RAGuard ZKIP reference implementations
- `eval/metrics.py:evaluate_run()` — ASR, Recall@5, F1, EM, latency
- `eval/analyze_failures.py` — error analysis
- `scripts/` — all CLI entry points; `config.py` — every hyperparameter

## Configuration

Everything tunable lives in `config.py`. Critical values:
- `STAGE1_THRESHOLD = 0.5` — τ₁, tune via F2 score on validation set
- `CONTRADICTION_THRESHOLD = 0.7` — τ₃ for NLI stage
- `TOP_K_RETRIEVAL = 10`, `TOP_K_AFTER_TRUST = 5`
- Trust weights: alignment=0.35, classifier=0.35, coherence=0.30

## Data Format

Questions file (JSONL): `{"question": str, "answer": str, "gold_doc_id": int, "target_answer": str}` per line. `target_answer` is the attacker's desired output (only on poisoned-target queries). Documents file (JSONL): one document per line with `doc_id`.

## Critical Implementation Notes

1. **FAISS normalization:** `IndexFlatIP` requires L2-normalized embeddings before `add`/`search`. Skipping this silently breaks cosine similarity.
2. **DeBERTa tokenizer:** Use `AutoTokenizer`, not `BertTokenizer` — DeBERTa-v3 uses SentencePiece.
3. **NLI input length:** DeBERTa-v3-large NLI has a 512-token limit; truncate documents before pairwise comparison.
4. **Training-data isolation:** The 5,000 poisoned pairs for Stage 1 training MUST use queries disjoint from the 200 test target questions.
5. **ASR measurement:** Case-insensitive substring match (not exact match), following PoisonedRAG methodology.
6. **Memory budget:** Stage 1 (~350MB) + Stage 3 (~1.3GB) + Llama 4-bit (~6GB) ≈ 8GB. Load one large model at a time if GPU is constrained.

## Evaluation Targets

| Metric | Target |
|--------|--------|
| ASR | ≤ 0.05 |
| Recall@5 degradation | ≤ 3% vs. clean baseline |
| Latency overhead | ≤ 33% of RAGuard's ZKIP |

## Ablation Study (Required)

Record ASR + Recall@5 for: Stage 1 only → Stages 1+2 → Full pipeline (1+2+3), at each poison ratio (5%, 10%, 20%), on NQ and HotpotQA.

## Additional Context

- `handoff.md` — full project handoff (architecture rationale, paper extension, team context)
- `stage2_handoff.md` — Stage 1 → Stage 2 transition notes, training results, model artifact location
