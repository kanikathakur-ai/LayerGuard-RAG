# Stage 2 Handoff: Trust Scoring

**From:** Stage 1 (classifier)
**To:** Stage 2 (trust scoring + re-ranking)
**Date:** 2026-05-12

---

## What Stage 1 Did

Stage 1 fine-tuned `microsoft/deberta-v3-base` as a binary cross-encoder that scores every retrieved document with P(poisoned | query, document). Documents above threshold τ₁ are dropped before Stage 2 sees them.

### Training results

| Split | F2 score | Loss |
|-------|----------|------|
| Validation (epoch 2) | 1.000 | 1.53e-05 |
| Validation (epoch 3) | 1.000 | 1.15e-05 |
| **Test (held-out)** | **1.000** | — |

Training: 8,000 examples, 3 epochs, ~15 min on RTX 3090, `train_loss=0.0074`.
Note: F2=1.0 reflects easy synthetic training data (template-generated poison docs). Real-world F2 on the NQ test set at 5–20% poison ratios is the meaningful number — that requires the full pipeline evaluation in `scripts/run_experiments.py`.

---

## Files You Need

| Path | Description |
|------|-------------|
| `results/stage1_classifier/` | Trained model + tokenizer — load from HF (see below) |
| `data/nq/documents.jsonl` | 50K Wikipedia passages — shared corpus |
| `data/nq/faiss.index` + `faiss_embeddings.npy` | Pre-built FAISS index over corpus |
| `data/nq/test_questions.jsonl` | 500 NQ questions; 239/500 have `gold_doc_id` set |
| `src/defense/stage1_classifier.py` | Stage 1 module |
| `src/defense/stage2_trust.py` | Stage 2 stub — your file to implement |
| `config.py` | All hyperparameters |

> **Large files not in git** — `data/` and `results/` are gitignored due to size.
> The trained Stage 1 model is on HuggingFace: **`michchicken/layerguard-stage1`**
> Data and index are on nlp-gpu-01 at `/home/mjsheu/NLP203/data/`, or regenerate with the setup scripts below.

---

## Stage 1 Interface — What You Receive

Stage 1 hands your code a list of surviving documents and their P(poisoned) scores.

```python
from src.defense.stage1_classifier import load_classifier, filter_documents

model, tokenizer = load_classifier("michchicken/layerguard-stage1")  # loads from HuggingFace
# or locally if you have it: load_classifier("results/stage1_classifier")

# For each query + its top-10 retrieved docs:
surviving_docs, scores = filter_documents(
    query="who plays jon snow in game of thrones",
    documents=retrieved_docs,   # list[str], top-10 from FAISS
    model=model,
    tokenizer=tokenizer,
    threshold=0.5,              # τ₁ — default in config.py
)
# surviving_docs: list[str]  — docs where P(poisoned) ≤ τ₁
# scores:         list[float] — P(poisoned) for each survivor (= 1 - classifier_confidence)
```

The scores feed directly into your trust formula as `cᵢ = 1 - scores[i]`.

---

## What Stage 2 Must Implement

**File:** `src/defense/stage2_trust.py` (stub already exists)

Trust score formula: **T(dᵢ) = 0.35·aᵢ + 0.35·cᵢ + 0.30·hᵢ**

| Component | Symbol | How to compute |
|-----------|--------|----------------|
| Semantic alignment | `aᵢ` | Avg cosine sim of doc's MiniLM embedding to all other survivors' embeddings |
| Classifier confidence | `cᵢ` | `1 - P(poisoned)` — comes from Stage 1 scores above |
| Query-doc coherence | `hᵢ` | Inverse suspicion: cosine sim between query embedding and doc embedding |

Re-rank survivors by T(dᵢ), keep top-5 (`TOP_K_AFTER_TRUST = 5` in config.py).

Functions to implement (stubs in `stage2_trust.py`):
```python
compute_semantic_alignment(doc_embeddings: np.ndarray) -> list[float]
compute_trust_scores(doc_embeddings, classifier_scores, weights) -> list[float]
rerank_and_filter(documents, trust_scores, top_k=5) -> list[tuple[str, float]]
```

MiniLM embeddings are already available from the retriever — no new model to load.

---

## Full Pipeline Context

```
Retriever (top-10) → Stage 1 filter → Stage 2 re-rank (top-5) → Stage 3 NLI → LLM
```

The pipeline orchestrator is `src/defense/pipeline.py:defend_and_answer()`. It already has Stage 1 wired in; you'll slot Stage 2 in between the Stage 1 call and the Stage 3 call.

---

## Regenerating Data/Model (if needed)

```bash
# From /home/mjsheu/NLP203, using the conda base Python:
python scripts/prepare_nq.py                     # ~30s: downloads corpus + questions
python scripts/build_index.py                    # ~6 min: builds FAISS index
python scripts/gen_stage1_data.py                # ~30s: generates training data
CUDA_VISIBLE_DEVICES=3 ~/miniconda3/bin/python scripts/train_stage1.py \
    --train-data data/synthetic_train/train.jsonl \
    --val-data   data/synthetic_train/val.jsonl \
    --output-dir results/stage1_classifier      # ~15 min on RTX 3090
```

---

## Hyperparameters (from `config.py`)

```python
STAGE1_THRESHOLD      = 0.5    # τ₁ — tune after Stage 2 is integrated
TRUST_WEIGHT_ALIGNMENT = 0.35  # wₐ
TRUST_WEIGHT_CLASSIFIER = 0.35 # w_c
TRUST_WEIGHT_COHERENCE  = 0.30 # w_h
TOP_K_AFTER_TRUST     = 5
```

Threshold τ₁ should be tuned jointly with Stage 2 weights via `scripts/tune_thresholds.py` once the full pipeline is wired up.

---

## Known Limitations / Watch-outs

- **gold_doc_id coverage is 47.8%** (239/500 test questions): the BeIR/nq corpus sample doesn't contain the gold passage for every NQ question. Recall@5 is computed only over those 239 questions. This is acceptable for ablation but worth noting.
- **Synthetic poison docs are easy**: the template-based attack in `inject_poison.py` produces highly detectable text. Stage 1 F2=1.0 will drop once tested against more realistic attacks. The ablation at 5/10/20% poison ratios in `run_experiments.py` is the real benchmark.
- **τ₁ is untuned**: the default 0.5 threshold was not grid-searched yet. Run `scripts/tune_thresholds.py` after your stage is done.
