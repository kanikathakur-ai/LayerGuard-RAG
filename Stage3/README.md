# Stage 3: Cross-Document Consistency Checker

## What it does
Stage 3 is the final stage of the LayerGuard-RAG defense pipeline.
It catches poisoned documents that Stages 1 and 2 missed by looking
at ALL retrieved documents together and flagging any document that
tells a different story from the majority.

**Example:** If 4 documents say "X bank had security breaches" and
1 document says "X bank is completely safe", Stage 3 flags and removes
the outlier.

## How it works
1. **Pass 1 — Cosine Similarity:** Converts all documents into embedding
   vectors using `all-MiniLM-L6-v2` and computes each document's average
   similarity to all others. Documents below the similarity threshold are
   flagged as outliers.
2. **Pass 2 — Trust Score Gating:** Flagged documents are only REMOVED if
   they also have a low trust score from Stage 2. A flagged document with
   a high trust score is kept.

## What it catches ✅
- Documents that contradict the general consensus of the retrieved set
- Coordinated multi-document poisoning (RAGuard's known failure mode)
- Medium-to-obvious narrative contradictions

## What it does NOT catch ❌
- Subtle single-fact poisoning (e.g. wrong date, wrong number)
- Those cases are handled by Stages 1 and 2

## Requirements
```bash
pip install sentence-transformers torch
```

## How to run Stage 3 standalone
```bash
python stage3.py
```
This runs the built-in bank example to verify everything is working.

## How to import into the pipeline
```python
from stage3 import stage3_consistency_check

# After Stage 2 produces documents and trust scores:
clean_docs, clean_scores, removed = stage3_consistency_check(
    documents=stage2_documents,    # list of 5 document strings
    trust_scores=stage2_trust_scores  # list of 5 floats between 0 and 1
)

# Pass clean_docs to the LLM generator
```

## Parameters
| Parameter | Default | Description |
|---|---|---|
| `documents` | required | list of document strings from Stage 2 |
| `trust_scores` | required | list of float trust scores from Stage 2 |
| `similarity_threshold` | 0.62 | documents with avg cosine similarity below this are flagged |
| `trust_threshold` | 0.5 | flagged documents below this trust score are removed |

## Output
| Output | Type | Description |
|---|---|---|
| `clean_documents` | list of strings | filtered documents to pass to LLM |
| `clean_trust_scores` | list of floats | corresponding trust scores |
| `removed_indices` | list of ints | indices of documents that were removed |

## Test results
| Test | Result | Description |
|---|---|---|
| Test 1: Bank example | ✅ PASSED | Obvious poisoning caught and removed |
| Test 2: All clean docs | ✅ PASSED | Nothing incorrectly removed |
| Test 3: High trust score | ✅ PASSED | Flagged doc correctly kept |
| Test 4: Subtle poisoning | ❌ EXPECTED FAIL | Single wrong date — too subtle, handled by Stages 1 & 2 |
| Test 5: Medium subtlety | ✅ PASSED | Medium contradiction caught and removed |

## Files
- `stage3.py` — main implementation
- `test_stage3.py` — all 6 test cases

## Model used
- `all-MiniLM-L6-v2` (SentenceTransformers) — no fine-tuning needed
