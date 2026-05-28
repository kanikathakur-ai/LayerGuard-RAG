import json
import time
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, TOP_K_RETRIEVAL, TOP_K_AFTER_TRUST
from src.retriever import load_documents, load_index, retrieve
from src.defense.stage1_classifier import load_classifier, score_documents
from src.defense.stage2_trust import score_and_rerank_documents


DOCS_PATH = "data/nq/documents.jsonl"
QUESTIONS_PATH = "data/nq/test_questions.jsonl"
INDEX_PATH = "data/nq/faiss.index"
EMB_PATH = "data/nq/faiss_embeddings.npy"
STAGE1_MODEL_PATH = "michchicken/layerguard-stage1"
OUTPUT_PATH = "results/stage2_metrics.json"


def load_questions(path):
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def main():
    print("Loading documents...")
    documents = load_documents(DOCS_PATH)

    print("Loading FAISS index...")
    index, _ = load_index(INDEX_PATH, EMB_PATH)

    print("Loading embeddings...")
    doc_embeddings = np.load(EMB_PATH)

    print("Loading MiniLM encoder...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading Stage 1 classifier...")
    stage1_model, stage1_tokenizer = load_classifier(STAGE1_MODEL_PATH)

    questions = load_questions(QUESTIONS_PATH)

    total_questions = 0
    recall_questions = 0
    recall_hits = 0
    total_stage1_survivors = 0
    total_stage2_docs = 0
    total_stage2_time = 0.0
    examples = []

    for q_idx, q in enumerate(questions):
        query = q["question"]
        gold_doc_id = q.get("gold_doc_id")

        retrieved = retrieve(query, index, documents, encoder, k=TOP_K_RETRIEVAL)

        ret_docs = [doc for doc, _, _ in retrieved]
        ret_ids = [doc_id for _, doc_id, _ in retrieved]
        ret_scores = [score for _, _, score in retrieved]

        poison_scores = score_documents(query, ret_docs, stage1_model, stage1_tokenizer)

        survivors = [
            (doc, poison_score, ret_score, doc_id)
            for doc, poison_score, ret_score, doc_id
            in zip(ret_docs, poison_scores, ret_scores, ret_ids)
            if poison_score <= 0.5
        ]

        if not survivors:
            min_idx = min(range(len(poison_scores)), key=lambda i: poison_scores[i])
            survivors = [(ret_docs[min_idx], poison_scores[min_idx], ret_scores[min_idx], ret_ids[min_idx])]

        s1_docs, s1_scores, s1_ret_scores, s1_ids = zip(*survivors)
        s1_embs = np.array([doc_embeddings[doc_id] for doc_id in s1_ids])

        t0 = time.perf_counter()
        stage2_ranked = score_and_rerank_documents(
            documents=list(s1_docs),
            doc_embeddings=s1_embs,
            classifier_scores=list(s1_scores),
            retrieval_scores=list(s1_ret_scores),
            top_k=TOP_K_AFTER_TRUST,
        )
        stage2_time = time.perf_counter() - t0

        stage2_doc_ids = []
        for item in stage2_ranked:
            doc_text = item["document"]
            for doc, _, _, doc_id in survivors:
                if doc == doc_text:
                    stage2_doc_ids.append(doc_id)
                    break

        total_questions += 1
        total_stage1_survivors += len(s1_docs)
        total_stage2_docs += len(stage2_ranked)
        total_stage2_time += stage2_time

        if gold_doc_id is not None:
            recall_questions += 1
            if gold_doc_id in stage2_doc_ids:
                recall_hits += 1

        if len(examples) < 10:
            examples.append({
                "question": query,
                "gold_doc_id": gold_doc_id,
                "stage1_survivors": len(s1_docs),
                "stage2_doc_ids": stage2_doc_ids,
                "stage2_trust_scores": [item["trust_score"] for item in stage2_ranked],
                "stage2_latency_s": stage2_time,
            })

        if (q_idx + 1) % 50 == 0:
            print(f"Processed {q_idx + 1}/{len(questions)} questions...")

    metrics = {
        "total_questions": total_questions,
        "questions_with_gold_doc_id": recall_questions,
        "stage2_recall_at_5": recall_hits / recall_questions if recall_questions else None,
        "recall_hits": recall_hits,
        "avg_stage1_survivors": total_stage1_survivors / total_questions,
        "avg_stage2_docs_returned": total_stage2_docs / total_questions,
        "avg_stage2_latency_s": total_stage2_time / total_questions,
    }

    os.makedirs("results", exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"metrics": metrics, "examples": examples}, f, indent=2)

    print("\nStage 2 metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nSaved Stage 2 metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
