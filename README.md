# LayerGuard-RAG

**LayerGuard-RAG** is a three-stage defense pipeline that protects Retrieval-Augmented Generation (RAG) systems against data-poisoning attacks (specifically PoisonedRAG). It sits between the retriever and the LLM generator and filters poisoned documents before the LLM sees them. The project extends [RAGuard (Kolhe et al., 2025)](https://arxiv.org/abs/2501.05553), replacing their expensive ZKIP mechanism (k+1 LLM forward passes per query) with three cheaper, complementary stages.

```
User Query → Retriever (top-10, FAISS + MiniLM) →
  Stage 1: DeBERTa-v3-base cross-encoder — drop if P(poisoned) > τ₁ →
  Stage 2: Trust scoring — re-rank by T(dᵢ) = 0.35·aᵢ + 0.35·cᵢ + 0.30·hᵢ, keep top-5 →
  Stage 3: NLI contradiction graph (DeBERTa-v3-large) — drop docs contradicting ≥2 peers →
  Llama-3.1-8B-Instruct (4-bit) → Answer
```

For full architecture rationale and design decisions, see [`handoff.md`](handoff.md).
For Stage 3 evaluation results, see [`stage_3_jincheng/RESULTS.md`](stage_3_jincheng/RESULTS.md).

---

## Hardware Requirements

- **GPU (recommended):** CUDA-capable GPU with ≥8 GB VRAM. Stage 1 (~350 MB) + Stage 3 (~1.3 GB) + Llama-3.1-8B 4-bit (~6 GB) ≈ 8 GB total.
- **CPU fallback:** All stages run on CPU; generation is very slow without a GPU.
- **macOS note:** faiss-cpu + torch both load libomp and can segfault. Prefix all commands with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`.

---

## Environment Setup

Requires **Python ≥ 3.11**.

### Option A — uv (recommended)

```bash
pip install uv          # if not already installed
uv sync                 # installs all deps from uv.lock
```

### Option B — pip

```bash
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

> **CPU-only:** replace `torch==2.12.0+cu126` in `requirements.txt` with `torch==2.12.0` before installing.

### HuggingFace authentication

Two models require HuggingFace access:

- **Llama-3.1-8B-Instruct** (`meta-llama/Llama-3.1-8B-Instruct`) — gated; request access at [huggingface.co/meta-llama](https://huggingface.co/meta-llama).
- **Mistral-7B-Instruct-v0.3** (`mistralai/Mistral-7B-Instruct-v0.3`) — automatic fallback if Llama is unavailable.

Authenticate once:
```bash
huggingface-cli login
# or: export HF_TOKEN=<your_token>
```

> **Note on published results:** the experiments in `stage_3_jincheng/RESULTS.md` were run on **Mistral-7B-Instruct-v0.3** because Llama-3.1-8B was gated on that machine. Re-running on Llama may produce slightly different absolute numbers; trends and conclusions should hold.

---

## Downloading Data and Models

Datasets, FAISS indices, and model checkpoints are **not included in the submission** due to size. Download them before running experiments.

### NQ corpus, FAISS index, and test questions

```bash
python scripts/fetch_data.py
```

This downloads from [`michchicken/layerguard-nq`](https://huggingface.co/datasets/michchicken/layerguard-nq) on HuggingFace into `data/`.

### Stage 1 classifier

The trained Stage 1 model is published at [`michchicken/layerguard-stage1`](https://huggingface.co/michchicken/layerguard-stage1) on HuggingFace. The evaluation scripts load it automatically by name — no manual download needed. To retrain it locally, see `scripts/train_stage1.py`.

### PoisonedRAG adversarial data

Pre-generated GPT-4 adversarial passages for NQ are included at `PoisonedRAG/results/adv_targeted_results/nq.json`. The `--attack poisonedrag` experiment step uses this file.

---

## Reproducing All Experimental Results

All experiments can be run with a single command from the repo root:

```bash
# Full run (~1–2 hrs on GPU):
python scripts/reproduce_all.py

# Fast smoke test (all steps, small sizes):
python scripts/reproduce_all.py --quick

# Skip data fetch if data/ is already populated:
python scripts/reproduce_all.py --skip-data

# Run only specific steps:
python scripts/reproduce_all.py --steps stage3,templates

# Override generator model:
python scripts/reproduce_all.py --generator-model mistralai/Mistral-7B-Instruct-v0.3
```

### Experiment → result file mapping

| Step | Result file | Description |
|------|-------------|-------------|
| `stage2` | `results/stage2_metrics.json` | Stage 2 trust-scoring metrics on NQ |
| `stage3` | `results/stage3_eval_full.json` | Main ablation: NQ, dose ∈ {1,3,5}, configs {none, stage1, stage12, full} |
| `poisonedrag` | `results/stage3_eval_poisonedrag.json` | PoisonedRAG GPT-4 adversarial attack eval |
| `templates` | `results/stage3_eval_template_{assertion,encyclopedia,narrative,news,textbook}_quick.json` | Template-generalization (5 poison templates) |

The ablation table from `stage_3_jincheng/RESULTS.md` is reproduced by the `stage3` step.

### Running individual experiments

```bash
# Stage 2 metrics
python eval_stage2_metrics.py

# Stage 3 main ablation
python stage_3_jincheng/run_eval.py \
    --output results/stage3_eval_full.json \
    --doses 1 3 5 \
    --configs none stage1 stage12 full

# PoisonedRAG adversarial attack
# --resolve-gold fills gold_doc_id via retrieval for ~91/100 PoisonedRAG targets
# (required for meaningful Recall@5; --generator-model matches the published run)
python stage_3_jincheng/run_eval.py \
    --attack poisonedrag \
    --adv-path PoisonedRAG/results/adv_targeted_results/nq.json \
    --resolve-gold \
    --generator-model mistralai/Mistral-7B-Instruct-v0.3 \
    --output results/stage3_eval_poisonedrag.json

# Template generalization (one template at a time)
python stage_3_jincheng/run_eval.py \
    --attack template --template assertion \
    --quick --output results/stage3_eval_template_assertion_quick.json
```

---

## Key Results (NQ, PoisonedRAG GPT-4 adversarial attack)

Results produced with Mistral-7B-Instruct-v0.3 on an RTX 6000 Ada GPU (2026-05-28).
See [`poisonedrag_eval_handoff.md`](poisonedrag_eval_handoff.md) for full analysis and
[`stage_3_jincheng/RESULTS.md`](stage_3_jincheng/RESULTS.md) for template-attack results and Stage 3 intrinsic metrics.

| Condition | Defense | ASR ↓ | Recall@5 |
|-----------|---------|-------|----------|
| Clean | none | 0.050 | 0.870 |
| Clean | full | 0.050 | 0.774 |
| Poison dose=1 | none | 0.450 | 0.826 |
| Poison dose=1 | stage1 | 0.450 | 0.826 |
| Poison dose=1 | stage12 | 0.420 | 0.713 |
| Poison dose=1 | **full** | 0.420 | 0.713 |
| Poison dose=3 | none | 0.670 | 0.713 |
| Poison dose=3 | stage1 | 0.670 | 0.713 |
| Poison dose=3 | stage12 | 0.700 | 0.626 |
| Poison dose=3 | **full** | 0.700 | 0.626 |
| Poison dose=5 | none | 0.940 | 0.261 |
| Poison dose=5 | stage1 | 0.940 | 0.261 |
| Poison dose=5 | stage12 | 0.970 | 0.200 |
| Poison dose=5 | **full** | 0.970 | 0.200 |

**Key findings:** Stage 1 (trained on template-style poison) provides **no protection** against GPT-4-generated adversarial text — ASR is unchanged from undefended at all doses. Stage 2 and Stage 3 also do not reduce ASR on this harder attack; Stage 2 marginally increases ASR at higher doses as it promotes topically-relevant poison docs. The primary next step is retraining Stage 1 on diverse/GPT-4-generated poison. **Stage 3 latency: ~100 ms/query**, vs RAGuard ZKIP's k+2 LLM forward passes.

---

## Repository Structure

```
LayerGuard-RAG/
├── config.py                       # all hyperparameters
├── eval/
│   ├── metrics.py                  # ASR, Recall@5, F1, EM, latency
│   └── analyze_failures.py         # error analysis
├── eval_stage2_metrics.py          # Stage 2 standalone eval
├── scripts/
│   ├── reproduce_all.py            # reproduce all experiments (this file)
│   ├── fetch_data.py               # download NQ corpus from HF
│   ├── prepare_nq.py               # prepare NQ corpus + test questions
│   ├── gen_stage1_data.py          # generate Stage 1 synthetic training data
│   ├── train_stage1.py             # train Stage 1 DeBERTa classifier
│   ├── tune_thresholds.py          # tune τ₁, τ₃, trust weights on val set
│   └── run_experiments.py          # vanilla RAG baseline
├── src/
│   ├── retriever.py                # FAISS + MiniLM retriever
│   ├── generator.py                # Llama/Mistral wrapper
│   ├── vanilla_rag.py              # undefended RAG baseline
│   ├── attacks/                    # PoisonedRAG injection + training-data gen
│   ├── baselines/                  # perplexity filter, RAGuard ZKIP reference
│   └── defense/
│       ├── stage1_classifier.py    # DeBERTa-v3-base cross-encoder
│       ├── stage2_trust.py         # trust scoring + reranking
│       ├── stage3_nli.py           # NLI contradiction graph
│       └── pipeline.py             # end-to-end orchestrator
├── stage_3_jincheng/               # Stage 3 implementation + eval harness
│   ├── run_eval.py                 # main evaluation script
│   ├── defended_pipeline.py        # retrieval → S1 → S2 → S3 → generator
│   ├── stage3_nli.py               # NLI module
│   ├── RESULTS.md                  # full evaluation results
│   └── README.md                   # Stage 3 design notes
├── requirements.txt                # pip-installable dependencies
├── pyproject.toml                  # uv/project metadata
├── handoff.md                      # full architecture rationale
└── stage2_handoff.md               # Stage 1→2 transition notes
```

### Caveats

- **Poison "dose" vs corpus ratio:** PoisonedRAG poison is injected as a fixed number of docs per query (not by corpus ratio). We ablate over dose ∈ {1, 3, 5} poison docs per target question.
- **Stage 1 does not generalise to GPT-4 poison:** The DeBERTa classifier was trained on template-style poison and learns surface patterns rather than semantic manipulation. It passes 100% of PoisonedRAG adversarial text — retraining on diverse/GPT-4 poison is the primary next step.
- **Stage 2 hurts under adversarial conditions:** The trust scorer promotes topically-relevant poison docs, which slightly increases ASR at higher doses (0.94 → 0.97 at dose=5). On clean queries it costs ~10 pts of Recall@5 (0.870 → 0.774).
- **Stage 3 does not fire on PoisonedRAG poison:** GPT-4-generated adversarial texts are written to be factually plausible and non-contradictory with peer documents, so the NLI contradiction graph finds nothing to remove.
- **Clean ASR noise floor is 0.05:** ~25% of PoisonedRAG target answers are very short (e.g. "2", "yes", "Paris") — substring match produces false positives on clean queries unrelated to the attack.
