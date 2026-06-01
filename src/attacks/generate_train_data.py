"""Generate synthetic (query, document, label) training data for Stage 1 classifier.

IMPORTANT: Uses held-out queries — these must be disjoint from the 200 test
target questions to avoid data leakage.
"""

import json
import os
import random

from tqdm import tqdm

from config import (
    SYNTHETIC_CLEAN_PAIRS,
    SYNTHETIC_POISONED_PAIRS,
    SYNTHETIC_TRAIN_DIR,
    SYNTHETIC_TRAIN_SPLIT,
    SYNTHETIC_VAL_SPLIT,
)
from src.attacks.inject_poison import generate_poison_docs_for_question


def generate_training_data(
    held_out_qa_pairs: list[
        dict
    ],  # [{"question": str, "answer": str, "gold_doc": str}]
    retrieved_clean_docs: dict,  # {question: [doc_text, ...]} from normal retrieval
    n_poisoned: int = SYNTHETIC_POISONED_PAIRS,
    n_clean: int = SYNTHETIC_CLEAN_PAIRS,
    seed: int = 0,
) -> list[dict]:
    """Build a balanced set of (query, document, label) examples.

    label=1 means poisoned, label=0 means clean.
    """
    rng = random.Random(seed)
    examples = []

    questions = [item["question"] for item in held_out_qa_pairs]
    rng.shuffle(questions)

    # Poisoned pairs
    q_pool = questions * ((n_poisoned // len(questions)) + 1)
    for i in range(n_poisoned):
        q = q_pool[i]
        target = f"fake answer {i}"  # placeholder; replace with realistic targets
        doc = generate_poison_docs_for_question(q, target, n_docs=1, seed=seed + i)[0]
        examples.append({"query": q, "document": doc, "label": 1})

    # Clean pairs
    clean_questions = [q for q in questions if q in retrieved_clean_docs]
    for i in range(n_clean):
        q = clean_questions[i % len(clean_questions)]
        docs = retrieved_clean_docs[q]
        if docs:
            doc = rng.choice(docs)
            examples.append({"query": q, "document": doc, "label": 0})

    rng.shuffle(examples)
    return examples


def split_and_save(examples: list[dict], output_dir: str = SYNTHETIC_TRAIN_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n = len(examples)
    n_train = int(n * SYNTHETIC_TRAIN_SPLIT)
    n_val = int(n * SYNTHETIC_VAL_SPLIT)

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train : n_train + n_val],
        "test": examples[n_train + n_val :],
    }
    for split_name, split_data in splits.items():
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(split_data)} examples to {path}")


def load_split(split: str, data_dir: str = SYNTHETIC_TRAIN_DIR) -> list[dict]:
    path = os.path.join(data_dir, f"{split}.jsonl")
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples
