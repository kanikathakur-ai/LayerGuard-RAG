# LayerGuard-RAG: Claude Code Handoff Document

## 1. Project Overview

**What this project is:** A three-stage defense pipeline that protects Retrieval-Augmented Generation (RAG) systems against data poisoning attacks. It sits between the retriever and the LLM generator in a standard RAG architecture.

**The problem:** An attacker can inject as few as 5 malicious documents into a knowledge base of millions and achieve 90–99% attack success rates against LLMs like GPT-4. Our system catches and removes those poisoned documents before the LLM ever sees them.

**Primary paper we extend:** RAGuard (Kolhe et al., 2025) — a two-stage defense using adversarial retriever training + leave-one-out inference filtering (ZKIP). We replace their expensive ZKIP mechanism (which requires k+1 full LLM forward passes per query) with three cheaper, complementary stages.

**Team:** 3 students, 4-week timeline. Starter code needed by Monday.

---

## 2. Architecture

```
User Query
    │
    ▼
┌──────────────────────┐
│  Retriever            │  all-MiniLM-L6-v2 embeddings + FAISS index
│  (top-k=10 docs)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  STAGE 1: Classifier  │  DeBERTa-v3-base cross-encoder
│  Binary filter         │  Input: [query; document] → P(poisoned)
│  Remove if P > τ₁      │  Threshold τ₁ tuned via F2 score
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  STAGE 2: Trust Score │  T(dᵢ) = 0.35·aᵢ + 0.35·cᵢ + 0.30·hᵢ
│  Re-rank & keep top-5  │  aᵢ = semantic alignment (avg cosine sim to peers)
│                        │  cᵢ = classifier confidence (1 - P(poisoned))
│                        │  hᵢ = query-doc coherence (inverse suspicion)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  STAGE 3: NLI Checker │  DeBERTa-v3-large (MNLI/FEVER/ANLI fine-tuned)
│  Contradiction graph   │  Pairwise NLI on top-5 → 10 pairs
│  Flag majority outliers│  Remove if contradicts ≥2 other docs
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LLM Generator        │  Llama-3.1-8B-Instruct (4-bit quantized)
│  Final answer          │
└──────────────────────┘
```

---

## 3. Repository Structure (Target)

```
layerguard-rag/
├── handoff.md                  # This file
├── README.md
├── requirements.txt
├── config.py                   # All hyperparameters, paths, thresholds
├── data/
│   ├── nq/                     # Natural Questions corpus (~50K docs)
│   ├── hotpotqa/               # HotpotQA corpus (stretch)
│   ├── poisoned/               # Generated poisoned documents
│   └── synthetic_train/        # Training data for Stage 1 classifier
├── src/
│   ├── __init__.py
│   ├── retriever.py            # FAISS indexing + retrieval
│   ├── generator.py            # Llama-3.1-8B-Instruct wrapper
│   ├── vanilla_rag.py          # End-to-end baseline RAG (no defense)
│   ├── defense/
│   │   ├── __init__.py
│   │   ├── stage1_classifier.py    # DeBERTa cross-encoder filter
│   │   ├── stage2_trust.py         # Trust scoring + re-ranking
│   │   ├── stage3_nli.py           # NLI contradiction graph
│   │   └── pipeline.py             # Full 3-stage pipeline orchestrator
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── inject_poison.py        # PoisonedRAG black-box attack wrapper
│   │   └── generate_train_data.py  # Synthetic data for Stage 1 training
│   └── baselines/
│       ├── perplexity_filter.py    # GPT-2 perplexity baseline
│       └── raguard_zkip.py         # RAGuard ZKIP reproduction (if feasible)
├── scripts/
│   ├── build_index.py          # One-time FAISS index construction
│   ├── train_stage1.py         # Fine-tune DeBERTa classifier
│   ├── tune_thresholds.py      # Grid search for τ₁, τ₃, trust weights
│   └── run_experiments.py      # Full experiment runner
├── eval/
│   ├── metrics.py              # ASR, Recall@5, F1, EM, latency
│   └── analyze_failures.py     # Failure mode categorization
└── results/
    └── (experiment outputs, tables, plots)
```

---

## 4. Dependencies & Environment

### Python 3.10, conda environment

```
# requirements.txt
torch>=2.1.0
transformers>=4.36.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4          # Use faiss-gpu if GPU available
datasets>=2.16.0
huggingface-hub>=0.20.0
accelerate>=0.25.0
bitsandbytes>=0.42.0      # For 4-bit Llama quantization
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=1.5.0
tqdm>=4.65.0
```

### Models to download (HuggingFace)

| Model | Purpose | Size |
|-------|---------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | Document/query embeddings (384-dim) | ~80MB |
| `microsoft/deberta-v3-base` | Stage 1 classifier (fine-tune this) | ~350MB |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | Stage 3 NLI (use as-is, no fine-tuning) | ~1.3GB |
| `meta-llama/Llama-3.1-8B-Instruct` | LLM generator (4-bit quantized) | ~4.5GB quantized |
| `openai-community/gpt2` | Perplexity baseline only | ~500MB |

### Hardware requirements
- Minimum: 1x GPU with 16GB VRAM (e.g., T4, A4000)
- Recommended: 1x GPU with 24GB+ VRAM (e.g., A100, L4)
- Llama-3.1-8B in 4-bit needs ~6GB VRAM; DeBERTa-large needs ~3GB; leave room for batching

---

## 5. Datasets

### Primary: Natural Questions (NQ)
- **Corpus:** ~50,463 documents (Wikipedia passages), following Anichkov et al.'s setup
- **Test set:** 500 questions sampled from the open-domain QA split
- **Target questions for attack:** 200 of those 500
- **Poisoned docs:** 5 per target question = 1,000 total injected into the corpus
- **Poison ratios to test:** 5%, 10%, 20%

### Secondary: HotpotQA
- **Corpus:** Distractor setting, ~7,405 dev-set questions with associated passages
- **Purpose:** Tests Stage 3's value on multi-hop reasoning where distributed poisoning is most dangerous

### Stretch: MS-MARCO (~8.8M passages) — only if NQ + HotpotQA done early

### Synthetic training data (for Stage 1 classifier)
- 5,000 poisoned (query, document) pairs: generated using PoisonedRAG's code on **held-out queries** (disjoint from the 200 test targets)
- 5,000 clean (query, document) pairs: from normal retrieval on the same held-out queries
- Total: 10,000 labeled examples
- Split: 80% train, 10% validation, 10% test

---

## 6. Implementation Details per Component

### 6.1 Retriever (`src/retriever.py`)

```python
# Key specs:
# - Embedding model: all-MiniLM-L6-v2 (384 dimensions)
# - Index: FAISS IndexFlatIP (inner product after L2 normalization = cosine sim)
#   - Use IndexIVFFlat if corpus > 100K docs for speed
# - Retrieve top-k=10 documents per query
# - Return: list of (doc_text, doc_id, score) tuples
# - Must support injecting poisoned docs into the index for attack simulation
```

Core functions needed:
- `build_index(documents: list[str]) -> faiss.Index`
- `save_index(index, path)` / `load_index(path) -> faiss.Index`
- `retrieve(query: str, index, documents, k=10) -> list[tuple[str, int, float]]`
- `inject_documents(index, new_docs, existing_docs) -> updated_index, updated_docs`

### 6.2 Generator (`src/generator.py`)

```python
# Key specs:
# - Model: meta-llama/Llama-3.1-8B-Instruct
# - Quantization: 4-bit via bitsandbytes (BitsAndBytesConfig(load_in_4bit=True))
# - Prompt format: system + context documents + question → short factual answer
# - max_new_tokens: 100
# - Use the model's chat template for Llama-3.1-Instruct format
```

Core functions needed:
- `load_generator() -> (model, tokenizer)`
- `generate_answer(query: str, context_docs: list[str], model, tokenizer) -> str`

Prompt template:
```
Based on the following documents, answer the question with a short factual answer.

Documents:
{doc1}
{doc2}
...

Question: {query}
Answer:
```

### 6.3 Stage 1: Classifier (`src/defense/stage1_classifier.py`)

```python
# Key specs:
# - Base model: microsoft/deberta-v3-base
# - Task: binary sequence classification (num_labels=2)
# - Input format: tokenizer(query, document, truncation=True, max_length=512)
# - Output: P(poisoned | query, document) via softmax
# - Threshold τ₁: tuned on validation set to maximize F2 score
# - Training: 3-5 epochs, lr=2e-5, batch_size=16, binary cross-entropy loss
# - Documents with P(poisoned) > τ₁ are REMOVED from candidate set
```

Core functions needed:
- `train_classifier(train_data, val_data, output_dir)`
- `load_classifier(model_dir) -> model`
- `score_documents(query, documents, model, tokenizer) -> list[float]`  # returns P(poisoned) for each
- `filter_documents(query, documents, model, tokenizer, threshold) -> list[tuple[str, float]]`  # returns survivors + their scores

Training script (`scripts/train_stage1.py`):
- Load 10K synthetic examples from `data/synthetic_train/`
- Fine-tune DeBERTa-v3-base with HuggingFace Trainer
- Save best checkpoint by F2 on validation set
- Log train/val loss and F2 per epoch

### 6.4 Stage 2: Trust Scoring (`src/defense/stage2_trust.py`)

```python
# Key specs:
# - Input: surviving documents from Stage 1 + their embeddings + classifier scores
# - Trust score formula: T(dᵢ) = wₐ·aᵢ + w_c·cᵢ + w_h·hᵢ
#   - aᵢ = avg cosine similarity of dᵢ's embedding to all other surviving docs' embeddings
#   - cᵢ = 1 - P(poisoned) from Stage 1 classifier
#   - hᵢ = inverse of suspicion score (proxy for query-doc coherence)
# - Default weights: wₐ=0.35, w_c=0.35, w_h=0.30
# - Re-rank by trust score, keep top-5
# - No additional model loading needed — reuses MiniLM embeddings + Stage 1 scores
```

Core functions needed:
- `compute_semantic_alignment(doc_embeddings: np.ndarray) -> list[float]`
- `compute_trust_scores(doc_embeddings, classifier_scores, weights) -> list[float]`
- `rerank_and_filter(documents, trust_scores, top_k=5) -> list[tuple[str, float]]`

### 6.5 Stage 3: NLI Checker (`src/defense/stage3_nli.py`)

```python
# Key specs:
# - Model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
#   - Pre-trained, NO fine-tuning needed
#   - Outputs: entailment / neutral / contradiction with probabilities
# - Input: top-5 documents from Stage 2
# - For each pair (dᵢ, dⱼ): compute P(contradiction | dᵢ, dⱼ)
#   - 5 docs → C(5,2) = 10 pairwise comparisons
# - Contradiction threshold τ₃: default 0.7
# - Build contradiction graph: edge if P(contradiction) > τ₃
# - Flag doc as outlier if it has edges to ≥ ceil((k-1)/2) = 2 other docs
# - Remove flagged docs (those contradicting the majority)
# - ~50ms total on GPU for 10 NLI forward passes
```

Core functions needed:
- `load_nli_model() -> pipeline`
- `compute_pairwise_contradictions(documents, nli_pipeline) -> np.ndarray`  # 5x5 matrix
- `build_contradiction_graph(contradiction_matrix, threshold) -> dict`
- `flag_outliers(graph, min_contradictions=2) -> list[int]`  # indices of outlier docs
- `filter_outliers(documents, trust_scores, outlier_indices) -> list[str]`

### 6.6 Pipeline Orchestrator (`src/defense/pipeline.py`)

```python
# Glues everything together:
# 1. Retrieve top-10 with FAISS
# 2. Stage 1: classify + filter
# 3. Stage 2: trust score + re-rank to top-5
# 4. Stage 3: NLI check + remove outliers
# 5. Pass surviving docs to Llama generator
# 6. Return answer + metadata (which docs survived, scores, timing)
```

Core function:
- `defend_and_answer(query, retriever, generator, stage1_model, nli_model, config) -> dict`
  - Returns: `{"answer": str, "surviving_docs": list, "stage1_filtered": list, "stage3_flagged": list, "trust_scores": list, "timings": dict}`

---

## 7. Evaluation Metrics (`eval/metrics.py`)

### Attack Success Rate (ASR)
- Fraction of target queries where the generated answer contains the attacker's target answer
- Measured via case-insensitive substring matching (following PoisonedRAG)
- **Lower is better.** Target: ASR ≤ 0.05

### Recall@5
- Whether the gold (correct) document appears in the top-5 after defense filtering
- Target: degradation ≤ 3% relative to clean (no-poison) baseline

### F1 Score and Exact Match (EM)
- Standard QA metrics on NON-poisoned queries
- Ensures the defense doesn't break normal performance
- Use SQuAD-style F1 (token-level overlap) and EM (exact string match after normalization)

### Latency Overhead
- Wall-clock time per query for: Stage 1, Stage 2, Stage 3, full pipeline
- Compare against vanilla RAG and RAGuard's ZKIP
- Target: total overhead ≤ 33% of RAGuard's ZKIP

---

## 8. Baselines to Implement/Compare

| # | Baseline | Implementation | Notes |
|---|----------|---------------|-------|
| 1 | No Defense (vanilla RAG) | Implement ourselves | Upper bound on ASR |
| 2 | Perplexity filtering | Implement ourselves | GPT-2 perplexity per doc, threshold filter. Zou et al. report ASR >87% even with this. |
| 3 | RAGuard (ZKIP) | Reproduce from their GitHub if feasible, else use published numbers | ASR=0.000 on NQ but k+1 LLM passes per query |
| 4 | RobustRAG | Compare vs. published numbers | Isolate-then-aggregate strategy |
| 5 | SeCon-RAG | Compare vs. published numbers | Semantic filtering + conflict-free framework |

---

## 9. Experiment Configurations

### Attack setup
- Attack method: PoisonedRAG black-box (primary)
- Poisoned docs per target question: 5
- Poison ratios: {5%, 10%, 20%}
- Target questions: 200 per dataset

### Ablation study (critical)
Run and record ASR + Recall@5 for:
1. Stage 1 only
2. Stages 1 + 2
3. Full pipeline (Stages 1 + 2 + 3)

...at each poison ratio, on each dataset.

### Failure mode analysis
- Sample 50 cases where LayerGuard-RAG fails (poisoned doc survives all 3 stages)
- Categorize: Stage 1 miss / trust score too high / NLI didn't detect contradiction / doc genuinely aligned with clean docs
- Specifically test multi-hop distributed poisoning on HotpotQA

---

## 10. Stretch Goals (Only if Ahead of Schedule)

1. **White-box PoisonedRAG attack:** gradient-optimized retrieval sub-text. Test if defense still holds.
2. **Anichkov et al.'s prompt-injection active-database attack:** fundamentally different poisoning mechanism.
3. **MS-MARCO dataset:** 8.8M passages, web-scale noise.

---

## 11. External Code & Resources

| Resource | URL | Usage |
|----------|-----|-------|
| PoisonedRAG code | Check Zou et al. 2025 paper for GitHub link | Attack generation, black-box poisoned text creation |
| RAGuard code | Check Kolhe et al. 2025 paper for GitHub link | ZKIP baseline reproduction |
| DeBERTa-v3-base | `microsoft/deberta-v3-base` on HuggingFace | Stage 1 classifier (fine-tune) |
| DeBERTa-v3-large NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` on HuggingFace | Stage 3 NLI (use as-is) |
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` on HuggingFace | Embeddings for retrieval + Stage 2 |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace (gated, needs access request) | LLM generator |
| SQuAD eval script | Widely available (official SQuAD GitHub) | F1 / EM metric implementation |

---

## 12. Key Hyperparameters (All in `config.py`)

```python
# Retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K_RETRIEVAL = 10

# Stage 1
STAGE1_MODEL = "microsoft/deberta-v3-base"
STAGE1_MAX_LENGTH = 512
STAGE1_THRESHOLD = 0.5          # τ₁ — tune via F2 on val set
STAGE1_TRAIN_EPOCHS = 3
STAGE1_LEARNING_RATE = 2e-5
STAGE1_BATCH_SIZE = 16

# Stage 2
TRUST_WEIGHT_ALIGNMENT = 0.35   # wₐ
TRUST_WEIGHT_CLASSIFIER = 0.35  # w_c
TRUST_WEIGHT_COHERENCE = 0.30   # w_h
TOP_K_AFTER_TRUST = 5

# Stage 3
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
CONTRADICTION_THRESHOLD = 0.7   # τ₃
MIN_CONTRADICTIONS_TO_FLAG = 2  # ceil((k-1)/2) for k=5

# Generator
GENERATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GENERATOR_MAX_NEW_TOKENS = 100
GENERATOR_LOAD_IN_4BIT = True

# Attack
POISON_DOCS_PER_QUESTION = 5
POISON_RATIOS = [0.05, 0.10, 0.20]
NUM_TARGET_QUESTIONS = 200
NUM_TEST_QUESTIONS = 500

# Training data
SYNTHETIC_POISONED_PAIRS = 5000
SYNTHETIC_CLEAN_PAIRS = 5000
```

---

## 13. Week-by-Week Milestones

| Week | Deliverable | Success Criteria |
|------|-------------|-----------------|
| 0 (now→Mon) | Vanilla RAG + attack + eval harness | Can run query→retrieve→generate; PoisonedRAG attack achieves ASR >80%; metrics compute correctly |
| 1 | Stage 1 classifier trained and integrated | DeBERTa fine-tuned on 10K examples; threshold selected; Stage-1-only ASR measured |
| 2 | Stages 2 and 3 implemented | Trust scoring re-ranks docs; NLI checker flags outliers; each module has unit tests |
| 3 | Full pipeline integrated + NQ experiments + baselines | End-to-end `defend_and_answer()` works; all metrics on NQ at 3 poison ratios; perplexity baseline done; ablation table complete |
| 4 | HotpotQA experiments + failure analysis + report | HotpotQA results; 50-case failure categorization; comparison table vs. baselines; final report written |

---

## 14. Common Pitfalls & Notes

1. **Llama access is gated:** Request access on HuggingFace ahead of time. It can take hours to be approved. If blocked, use `mistralai/Mistral-7B-Instruct-v0.3` as a drop-in replacement.
2. **FAISS normalization:** When using `IndexFlatIP`, you MUST L2-normalize embeddings before adding/searching. Otherwise cosine similarity won't work correctly.
3. **DeBERTa tokenizer:** DeBERTa-v3 uses a SentencePiece tokenizer. Make sure you use `AutoTokenizer`, not `BertTokenizer`.
4. **NLI input length:** DeBERTa-v3-large NLI model has a 512-token limit. Truncate long documents before pairwise comparison.
5. **Held-out queries for training:** The 5,000 poisoned training pairs for Stage 1 MUST use different queries than the 200 test target questions. Otherwise you're leaking test data.
6. **PoisonedRAG corpus format:** Check their repo for the exact format of the NQ corpus they use. You need to match it exactly for the attack code to work.
7. **Memory management:** Load one large model at a time if GPU is tight. Stage 1 DeBERTa (~350MB) + Stage 3 DeBERTa-large (~1.3GB) + Llama 4-bit (~6GB) = ~8GB total, which fits on a 16GB GPU but leave room for activations.
8. **Evaluation consistency:** Use substring matching for ASR (case-insensitive), matching PoisonedRAG's methodology exactly. Don't use exact match for ASR.
