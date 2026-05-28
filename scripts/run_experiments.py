"""Full experiment runner: vanilla RAG baseline with poison injection.

Usage:
    python scripts/run_experiments.py \
        --docs data/nq/documents.jsonl \
        --questions data/nq/test_questions.jsonl \
        --poison-ratio 0.05 \
        --output results/vanilla_rag_5pct.json
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL, INDEX_PATH, POISON_DOCS_PER_QUESTION,
    NUM_TARGET_QUESTIONS, TOP_K_RETRIEVAL,
)
from src.retriever import load_documents, build_index, load_index
from src.generator import load_generator
from src.vanilla_rag import batch_vanilla_rag
from src.attacks.inject_poison import run_attack, rebuild_index_with_poison
from eval.metrics import evaluate_run, print_metrics


def load_questions(path: str) -> list[dict]:
    """Load questions from JSONL. Expects {"question", "answer", "gold_doc_id"} per line."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--poison-ratio", type=float, default=0.05)
    parser.add_argument("--n-target", type=int, default=NUM_TARGET_QUESTIONS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-poison", action="store_true", help="Run clean baseline")
    args = parser.parse_args()

    # Load corpus
    print("Loading corpus...")
    documents = load_documents(args.docs)
    questions = load_questions(args.questions)

    # Split target (to-be-poisoned) vs clean queries
    target_qs = questions[:args.n_target]
    clean_qs = questions[args.n_target:]

    encoder = SentenceTransformer(EMBEDDING_MODEL)

    if args.no_poison:
        print("Running clean baseline (no poison)...")
        index, embeddings = build_index(documents, encoder)
        eval_questions = questions
        poison_mask = [False] * len(questions)
        target_answers = [None] * len(questions)
    else:
        print(f"Generating poisoned documents (ratio={args.poison_ratio})...")
        # Use a wrong/adversarial answer for each target question
        attack_targets = [
            {"question": q["question"], "target_answer": q.get("target_answer", "unknown adversarial answer")}
            for q in target_qs
        ]
        poisoned_examples = run_attack(attack_targets, n_docs_per_question=POISON_DOCS_PER_QUESTION, seed=42)

        print(f"Rebuilding index with poison ratio {args.poison_ratio}...")
        contaminated_docs, poison_indices, index, embeddings = rebuild_index_with_poison(
            documents, poisoned_examples, args.poison_ratio, encoder
        )
        documents = contaminated_docs
        eval_questions = questions
        poison_mask = [True] * len(target_qs) + [False] * len(clean_qs)
        target_answers = [q.get("target_answer") for q in target_qs] + [None] * len(clean_qs)

    # Load generator
    print("Loading generator...")
    model, tokenizer = load_generator()

    # Run vanilla RAG
    queries = [q["question"] for q in eval_questions]
    results = batch_vanilla_rag(queries, index, documents, encoder, model, tokenizer, k=TOP_K_RETRIEVAL)

    # Evaluate
    gold_answers = [q["answer"] for q in eval_questions]
    gold_doc_ids = [q.get("gold_doc_id") for q in eval_questions]

    metrics = evaluate_run(results, gold_answers, target_answers, gold_doc_ids, poison_mask)
    print_metrics(metrics, label=f"Vanilla RAG (poison_ratio={args.poison_ratio})")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"metrics": metrics, "results": results[:10]}, f, indent=2, default=str)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
