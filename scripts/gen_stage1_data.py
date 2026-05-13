"""Generate synthetic training data for the Stage 1 classifier.

Requires (run in order):
  1. python scripts/prepare_nq.py
  2. python scripts/build_index.py
  3. python scripts/gen_stage1_data.py   ← this script

What it does:
  - Retrieves top-10 clean docs per held-out question (for negative examples)
  - Finds gold_doc_id for each test question (doc containing the answer substring)
  - Generates 10K (poisoned + clean) training pairs via generate_train_data.py
  - Updates data/nq/test_questions.jsonl with gold_doc_ids

Outputs:
  data/synthetic_train/train.jsonl
  data/synthetic_train/val.jsonl
  data/synthetic_train/test.jsonl
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import EMBEDDING_MODEL, INDEX_PATH, NQ_DIR, TOP_K_RETRIEVAL
from src.retriever import load_documents, load_index, retrieve
from src.attacks.generate_train_data import generate_training_data, split_and_save

GOLD_SEARCH_K = 50  # retrieve top-50 to maximize chance of finding the gold doc


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(items: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def find_gold_doc_id(answer: str, retrieved: list[tuple]) -> int | None:
    """Return doc_id of the first retrieved doc containing the answer substring."""
    answer_lower = answer.lower()
    for doc_text, doc_id, _ in retrieved:
        if answer_lower in doc_text.lower():
            return doc_id
    return None


def main():
    docs_path = os.path.join(NQ_DIR, "documents.jsonl")
    embeddings_path = INDEX_PATH.replace(".index", "_embeddings.npy")
    held_path = os.path.join(NQ_DIR, "held_out_questions.jsonl")
    test_path = os.path.join(NQ_DIR, "test_questions.jsonl")

    print("Loading corpus and FAISS index...")
    documents = load_documents(docs_path)
    index, _ = load_index(INDEX_PATH, embeddings_path)
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    # --- Held-out questions: retrieve clean docs for negative training examples ---
    held_out = load_jsonl(held_path)
    print(f"Retrieving clean docs for {len(held_out)} held-out questions...")
    retrieved_clean_docs: dict[str, list[str]] = {}
    for item in tqdm(held_out, desc="held-out retrieval"):
        q = item["question"]
        results = retrieve(q, index, documents, encoder, k=TOP_K_RETRIEVAL)
        retrieved_clean_docs[q] = [doc for doc, _, _ in results]

    # --- Test questions: fill in gold_doc_ids via answer-substring search ---
    test_qs = load_jsonl(test_path)
    print(f"Finding gold_doc_ids for {len(test_qs)} test questions...")
    n_found = 0
    for item in tqdm(test_qs, desc="gold doc search"):
        results = retrieve(item["question"], index, documents, encoder, k=GOLD_SEARCH_K)
        gold_id = find_gold_doc_id(item["answer"], results)
        item["gold_doc_id"] = gold_id
        if gold_id is not None:
            n_found += 1
    save_jsonl(test_qs, test_path)
    print(f"gold_doc_id found for {n_found}/{len(test_qs)} test questions "
          f"({100*n_found/len(test_qs):.1f}%); remainder set to null (skipped in Recall@5)")

    # --- Generate and save synthetic training data ---
    print("Generating synthetic training data...")
    examples = generate_training_data(held_out, retrieved_clean_docs)
    split_and_save(examples)
    print("Done. Training data written to data/synthetic_train/")
    print("Next: python scripts/train_stage1.py "
          "--train-data data/synthetic_train/train.jsonl "
          "--val-data data/synthetic_train/val.jsonl "
          "--output-dir results/stage1_classifier")


if __name__ == "__main__":
    main()
