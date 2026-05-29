# Stage 3 — Cross-Document Consistency Checker (fresh implementation)

NLI contradiction-graph implementation of LayerGuard-RAG Stage 3, plus the glue
to connect it to Stages 1–2 and an end-to-end evaluation harness. Self-contained
under this folder; reuses the upstream stages from `src/` (does **not** modify them
or the older `src/defense/stage3_nli.py` / `Stage3/`).

## Method (per the proposal)

Among the docs that survive Stages 1–2, score `P(contradiction | d_i, d_j)` with
DeBERTa-v3-large NLI (`config.NLI_MODEL`) in **both** directions (NLI is
directional; we take the max as an undirected edge weight). Build a contradiction
graph at threshold τ₃ (`config.CONTRADICTION_THRESHOLD`); flag a doc as an outlier
if it contradicts **≥ ⌈(n−1)/2⌉** peers, where `n` is the *actual* surviving count
(dynamic — adapts when Stage 1 over-filters).

**Trust gate (robust, by design).** Stage 2 min-max-normalizes trust *per query*,
so its raw value is relative, not absolute — a fixed numeric cutoff is meaningless.
So a flagged outlier is removed only if **both**: (1) it's in the bottom half by
Stage-2 trust *rank* within the set, and (2) its Stage-1 clean-confidence
`1 − P(poisoned)` is below `STAGE3_CLEAN_CONF_THRESHOLD` (absolute, transfers
across queries). High-rank or high-confidence flagged docs are kept (anti-over-filter).

## Files

| File | What |
|------|------|
| `device.py` | `get_device()` cuda>mps>cpu + `hf_pipeline_device()` |
| `stage3_nli.py` | the Stage 3 module (`load_nli`, `pairwise_contradictions`, `build_graph`, `flag_outliers`, `apply_stage3`) |
| `defended_pipeline.py` | `run_defended(...)`: retrieval → S1 → S2 → S3 → generator, ablatable via `defense ∈ {none,stage1,stage12,full}`, with doc-id tracking |
| `run_eval.py` | injects poison, runs the ablation, computes ASR / Recall@5 / F1 / EM / latency + intrinsic Stage-3 metrics → `results/stage3_eval.json` |
| `isolated_stage3_test.py` | generation-free characterization of Stage 3 on constructed contradictory / poison-majority sets |

## Run

```bash
# From repo root. Full run (NQ, 100 targets + 50 clean, dose ∈ {1,3,5}):
uv run python stage_3_jincheng/run_eval.py

# Tiny smoke / intrinsic-only:
uv run python stage_3_jincheng/run_eval.py --quick
uv run python stage_3_jincheng/run_eval.py --no-generate

# Isolated Stage-3 detection:
uv run python stage_3_jincheng/isolated_stage3_test.py
```

**macOS note:** faiss-cpu + torch both load libomp → segfault. Prefix local runs
with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`. Not needed on the Linux/CUDA box.

## Design notes / honest caveats

- **Poison "dose", not corpus ratio.** The template attack generates a fixed small
  number of poison docs, so a 5/10/20 % corpus ratio collapses to the same thing.
  What controls Stage 3's input is poison docs *per query*, so we ablate over
  dose ∈ {1,3,5} (minority → majority in the retrieved set). Poison is appended to
  the corpus (positions are irrelevant to content-based FAISS retrieval), keeping
  original `gold_doc_id` values valid.
- **Stage 1 dominates the single-template NQ attack** (it was trained on the same
  template), so Stage 3 typically sees an already-clean set and the `full` vs
  `stage12` delta is ≈0 on NQ. The isolated test is the primary demonstration that
  Stage 3 *functions*.
- **NLI vs injection-style poison.** PoisonedRAG injection text ("…X is the correct
  answer…") may read as *neutral* rather than *contradictory* to a gold fact, which
  can limit NLI recall. `run_eval.py` measures this directly (Stage-3 precision/recall
  on poison, clean-flag false positives).
