# Stage 3 — Evaluation Results

**Setup:** NQ, 100 poisoned-target + 50 clean questions. Retriever MiniLM/FAISS (50,463 docs).
Generator **Mistral-7B-Instruct-v0.3** (Llama-3.1-8B was gated → auto-fallback). Stage 1 =
`michchicken/layerguard-stage1` (DeBERTa-v3-base). Stage 3 NLI =
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`. GPU: RTX 6000 Ada.
Wall-clock 63 min (14 cells × 150 q with generation). Poison ablated by **dose** (poison docs
per target question); see `README.md` for why dose, not corpus ratio.

## End-to-end ablation

| condition | defense | ASR | Recall@5 | F1 (clean) |
|-----------|---------|-----|----------|------------|
| clean | none | 0.000 | 0.694 | 0.074 |
| clean | full | 0.000 | 0.639 | 0.071 |
| poison dose=1 | none | **0.160** | 0.694 | 0.074 |
| poison dose=1 | stage1 | 0.000 | 0.694 | 0.074 |
| poison dose=1 | stage12 | 0.000 | 0.625 | 0.071 |
| poison dose=1 | **full** | 0.000 | 0.625 | 0.071 |
| poison dose=3 | none | **0.280** | 0.667 | 0.074 |
| poison dose=3 | stage1 | 0.000 | 0.694 | 0.074 |
| poison dose=3 | stage12 | 0.000 | 0.625 | 0.071 |
| poison dose=3 | **full** | 0.000 | 0.625 | 0.071 |
| poison dose=5 | none | **0.430** | 0.472 | 0.074 |
| poison dose=5 | stage1 | 0.000 | 0.694 | 0.074 |
| poison dose=5 | stage12 | 0.000 | 0.625 | 0.071 |
| poison dose=5 | **full** | 0.000 | 0.625 | 0.071 |

**Reading it:**
- Attack works and is dose-responsive: vanilla ASR **0.16 → 0.28 → 0.43** (dose 1→3→5);
  Recall@5 collapses to 0.47 at dose=5 (poison majority pushes the gold doc out of top-5).
- **Stage 1 alone** drives ASR → 0 at every dose (it was trained on this exact poison template)
  and *restores* Recall@5 (e.g. dose=5: 0.47 → 0.69 by removing the poison bloc).
- **`full` ≡ `stage12`** in every row → Stage 3 changed nothing end-to-end.
- **Stage 2 cost:** Recall@5 0.694 → 0.625 (per-query trust normalization demotes gold docs);
  Stage 3 adds **no** further loss.

## Intrinsic Stage-3 metrics (full cells)

| cell | poison→S3 | flagged | removed | poison_removed | clean_removed | gold kept | S3 latency |
|------|-----------|---------|---------|----------------|---------------|-----------|------------|
| clean | 0 | 145 | 0 | 0 | 0 | 34/100 | 102 ms/q |
| dose=1 | 0 | 140 | 0 | 0 | 0 | 33 | 100 ms/q |
| dose=3 | 0 | 154 | 0 | 0 | 0 | 33 | 101 ms/q |
| dose=5 | 0 | 138 | 0 | 0 | 0 | 33 | 100 ms/q |

- **0 poison ever reached Stage 3** (Stage 1 removed 100%), so `removed = 0` everywhere.
- Stage 3 *flagged* 138–154 docs/cell (NLI over-fires on legit paraphrases), but the trust-gate
  **removed none and lost 0 gold docs** → "inert but safe": no collateral damage end-to-end.
- **Latency ≈ 100 ms/query** (20 directed NLI passes for k≤5), vs ~1750 ms generation — ~5%
  overhead, vs RAGuard's k+1 full LLM passes (~6×).

## Isolated Stage-3 test (poison fed directly, large NLI)

| scenario | flagged | removed (after gate) | after-gate precision / recall |
|----------|---------|----------------------|-------------------------------|
| A. poison minority (1/5) | [P,C,C] | **[P]** | 1.0 / 1.0 |
| B. poison majority (3/5) | all 5 | [P,P] | 1.0 / 0.67 |
| C. all clean (5) | [] | [] | — (no over-filter) |
| D. injection vs fact (2) | [C,P] | **[P]** | 1.0 / 1.0 |

- **Stage 3 functions when poison reaches it:** minority poison and fact-vs-injection → poison
  removed, gold kept. The large NLI rates the template injection as a **contradiction (1.00)** —
  the earlier "NLI sees injection as neutral" worry does **not** hold for the large model.
- NLI over-flags clean paraphrases (e.g. "released June 2007" vs "announced January 2007" →
  0.99); the **trust-gate is what prevents those false flags from removing good docs**.
- **Poison-majority is the weak spot** (removed 2/3): when poison is the majority bloc the
  majority-vote graph degrades — the documented limitation — but still no gold lost.

## Bottom line

Stage 3 is correct and cheap (~100 ms), and the trust-gate makes it safe (zero collateral
damage). On NQ with this single-template attack it is **redundant**: Stage 1 (trained on the
same template) removes all poison upstream, so Stage 3 has nothing to act on. Stage 3's value
requires poison that survives Stage 1 — a harder/varied attack or multi-hop distributed poison
(HotpotQA), which is the natural next step. Separately, **Stage 2's per-query normalization
costs ~7 pts of Recall@5** and is the only stage that hurt clean retrieval here.
