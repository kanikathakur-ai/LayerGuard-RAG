"""Download and prepare NQ data for LayerGuard-RAG.

Sources:
  - Corpus: BeIR/nq (2.68M Wikipedia passages), we stream the first N_CORPUS
  - Questions: nq_open validation split for test, train split for held-out
  - Answers: nq_open (first answer per question)
  - gold_doc_id is set to null here; gen_stage1_data.py fills it in via retrieval.

Outputs:
  data/nq/documents.jsonl         — {"doc_id": int, "text": str}
  data/nq/test_questions.jsonl    — {"question": str, "answer": str, "gold_doc_id": null, "target_answer": str}
  data/nq/held_out_questions.jsonl — {"question": str, "answer": str}
"""

import argparse
import json
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import load_dataset
from tqdm import tqdm
from config import NQ_DIR

N_CORPUS = 50_000
N_TEST = 500
N_HELD_OUT = 300
SEED = 42


def stream_corpus(n: int) -> list[str]:
    """Stream first n passages from BeIR/nq corpus."""
    print(f"Streaming {n:,} passages from BeIR/nq corpus...")
    corpus = load_dataset("BeIR/nq", "corpus", split="corpus", streaming=True)
    passages = []
    for ex in tqdm(corpus, total=n, desc="corpus"):
        title = ex.get("title", "").strip()
        text = ex.get("text", "").strip()
        if not text:
            continue
        # Prepend title — mirrors standard DPR preprocessing
        passages.append(f"{title}. {text}" if title else text)
        if len(passages) >= n:
            break
    return passages


def stream_nq_open(split: str, n: int) -> list[dict]:
    """Stream first n examples from nq_open split, returning question/answer dicts."""
    print(f"Streaming {n} examples from nq_open/{split}...")
    ds = load_dataset("nq_open", split=split, streaming=True)
    items = []
    for ex in tqdm(ds, total=n, desc=f"nq_open/{split}"):
        answers = ex["answer"]
        if not answers:
            continue
        answer = answers[0] if isinstance(answers, list) else answers
        items.append({"question": ex["question"], "answer": answer})
        if len(items) >= n:
            break
    return items


def assign_target_answers(questions: list[dict], seed: int = SEED) -> None:
    """Set target_answer on each question to a different question's answer (in-place)."""
    rng = random.Random(seed)
    answers = [q["answer"] for q in questions]
    shuffled = answers.copy()
    for _ in range(100):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(shuffled, answers)):
            break
    for q, fake in zip(questions, shuffled):
        q["target_answer"] = fake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-corpus", type=int, default=N_CORPUS)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--n-held-out", type=int, default=N_HELD_OUT)
    args = parser.parse_args()

    os.makedirs(NQ_DIR, exist_ok=True)

    # 1. Corpus
    passages = stream_corpus(args.n_corpus)
    docs_path = os.path.join(NQ_DIR, "documents.jsonl")
    with open(docs_path, "w") as f:
        for doc_id, text in enumerate(passages):
            f.write(json.dumps({"doc_id": doc_id, "text": text}) + "\n")
    print(f"Wrote {len(passages):,} documents to {docs_path}")

    # 2. Test questions (gold_doc_id filled in later by gen_stage1_data.py)
    test_qs = stream_nq_open("validation", args.n_test)
    assign_target_answers(test_qs)
    for q in test_qs:
        q["gold_doc_id"] = None

    test_path = os.path.join(NQ_DIR, "test_questions.jsonl")
    with open(test_path, "w") as f:
        for q in test_qs:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(test_qs)} test questions to {test_path}")

    # 3. Held-out questions for Stage 1 training data generation
    held_out = stream_nq_open("train", args.n_held_out)
    held_path = os.path.join(NQ_DIR, "held_out_questions.jsonl")
    with open(held_path, "w") as f:
        for q in held_out:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(held_out)} held-out questions to {held_path}")
    print("Done. Next: python scripts/build_index.py && python scripts/gen_stage1_data.py")


if __name__ == "__main__":
    main()
