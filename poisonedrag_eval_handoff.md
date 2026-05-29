# PoisonedRAG Eval Handoff

**Date:** 2026-05-28  
**Session goal:** Integrate the real PoisonedRAG attack into LayerGuard's eval and run the full ablation.

---

## What was implemented this session

### Problem fixed
The original eval attack (`src/attacks/inject_poison.py`) used a fixed string template to generate poison docs. Stage 1's training data was generated from the **same template** (`src/attacks/generate_train_data.py:47`), so Stage 1 memorized surface patterns and removed 100% of poison — making the full pipeline look like `full ≡ stage12 ≡ stage1` in every result row. Not a meaningful test of Stage 2 or Stage 3.

### Solution
Added a new loader (`src/attacks/poisonedrag_attack.py`) that reads the **official PoisonedRAG GPT-4-generated poison** from the vendored repo (`PoisonedRAG/results/adv_targeted_results/nq.json` — 100 NQ target questions × 5 adversarial passages each). These have no fixed template and use plausible wrong answers (e.g. "24" vs true "23 episodes") — unlike LayerGuard's placeholder `target_answer` of unrelated entities.

The PoisonedRAG attack uses the `LM_targeted` construction: `question + ". " + adv_text` to guarantee retrieval without needing retriever access (black-box).

**Key constraint:** Only 10/100 PoisonedRAG NQ questions overlap with LayerGuard's original `data/test_questions.jsonl`. So the PoisonedRAG run uses **PoisonedRAG's 100 questions as the target set**, not LayerGuard's. `--resolve-gold` fills in `gold_doc_id` via retrieval (~91/100 found in the 50k corpus).

### Files changed
- **New:** `src/attacks/poisonedrag_attack.py` — loader (`load_poisonedrag`, `poisonedrag_questions`, `resolve_gold_doc_ids`)
- **Modified:** `stage_3_jincheng/run_eval.py` — `--attack {template,poisonedrag}`, `--adv-path`, `--resolve-gold` flags; incremental JSON writes after each cell; `status`/`cells_total`/`finished_at` metadata
- **Modified:** `scripts/run_experiments.py` — same `--attack`/`--adv-path` flags
- **New:** `.claude/skills/monitor-experiments/` — skill for launching (detached tmux) + monitoring (health checker with partial-JSON support)

### torch fix (important)
`pyproject.toml` had no pytorch index → uv resolved `torch==2.12.0+cu130` from PyPI, which is incompatible with the cluster's CUDA 12.6 driver. Fixed by adding:
```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
```
Then `uv lock && uv sync` → `torch==2.12.0+cu126`, `cuda: True`. **Always use `CUDA_VISIBLE_DEVICES=N`** when launching (see below) — the cluster has 6 GPUs, GPU 0 had stale memory from a killed process.

---

## The experiment currently running

**Session:** tmux `pr` (attach: `tmux attach -t pr`, log: `tail -f results/logs/pr.log`)  
**Output:** `results/stage3_eval_poisonedrag.json` (written incrementally after each cell)  
**Monitor:** `python .claude/skills/monitor-experiments/check_experiment.py`

**Status as of 2026-05-28 ~21:15:** ✅ Running healthy — **cell 1/14**, GPU 1 at 100% util (8.0/24 GiB).
- Gold resolution: 91/100 targets found in corpus.
- First completed cell: `clean / none` → ASR=0.050, Recall@5=0.870, F1=0.075.

**Full command:**
```bash
CUDA_VISIBLE_DEVICES=1 uv run python stage_3_jincheng/run_eval.py \
  --attack poisonedrag \
  --adv-path PoisonedRAG/results/adv_targeted_results/nq.json \
  --resolve-gold \
  --generator-model mistralai/Mistral-7B-Instruct-v0.3 \
  --output results/stage3_eval_poisonedrag.json
```

**What it runs:** 14 cells — 2 clean baseline + 3 doses × 4 defense configs:

| condition | defense |
|-----------|---------|
| clean | none, full |
| poison dose=1 | none, stage1, stage12, full |
| poison dose=3 | none, stage1, stage12, full |
| poison dose=5 | none, stage1, stage12, full |

**Expected runtime:** ~60–70 min on RTX 3090 (same as template run at 63 min).  
**Expected result:** vanilla ASR clearly > 0 across doses; Stage 1 ASR no longer trivially 0 (Stage 1 no longer memorizes the attack); Stage 3 potentially shows non-zero delta vs Stage 12.

> **Smoke test:** No longer needed — full run started and is healthy. If the full run dies, relaunch on any free GPU (2–5 are idle at ~284 MiB each); replace `CUDA_VISIBLE_DEVICES=1` accordingly.

---

## Running a smaller experiment

To smoke-test first (3 targets, 2 clean, dose=5 only, ~4–5 min):
```bash
# Check free GPU first
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Launch smoke test (replace N with a free GPU index)
CUDA_VISIBLE_DEVICES=N uv run python stage_3_jincheng/run_eval.py \
  --attack poisonedrag \
  --adv-path PoisonedRAG/results/adv_targeted_results/nq.json \
  --resolve-gold \
  --generator-model mistralai/Mistral-7B-Instruct-v0.3 \
  --output results/stage3_eval_poisonedrag_quick.json \
  --quick
```

`--quick` sets `n_target=3, n_clean=2, doses=[5]` → 6 cells total.

Or run detached (recommended to survive disconnect):
```bash
bash .claude/skills/monitor-experiments/run_detached.sh smoke \
  "CUDA_VISIBLE_DEVICES=N uv run python stage_3_jincheng/run_eval.py \
   --attack poisonedrag \
   --adv-path PoisonedRAG/results/adv_targeted_results/nq.json \
   --resolve-gold --generator-model mistralai/Mistral-7B-Instruct-v0.3 \
   --output results/stage3_eval_poisonedrag_quick.json --quick"
```

---

## What to look for in results

The key question: does real PoisonedRAG poison survive Stage 1?

**Healthy signs:**
- `poison_doseN / none` ASR clearly > 0 — attack lands (poison is retrieved)
- `poison_doseN / stage1` ASR < vanilla but > 0 — Stage 1 catches some but not all
- `poison_doseN / full` ASR < `stage12` — Stage 3 contributing

**If Stage 1 still drives ASR to 0:** the GPT-4-generated poison still has detectable patterns — check `intrinsic.retrieved_poison` to confirm poison is being retrieved at all, and `intrinsic.stage1_survivor_poison` to see how much Stage 1 lets through.

**Recall@5** should stay near 0.69 on the clean baseline. A big drop signals Stage 2 collateral damage (expected ~7 pts from trust normalization, per the template-attack results).

Reference: template-attack results are in `results/stage3_eval_full.json`.
