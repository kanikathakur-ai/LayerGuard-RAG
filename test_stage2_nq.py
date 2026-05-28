import json
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


def load_questions(path, n=5):
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
            if len(questions) >= n:
                break
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

    print("Loading Stage 1 classifier from Hugging Face...")
    stage1_model, stage1_tokenizer = load_classifier(STAGE1_MODEL_PATH)

    questions = load_questions(QUESTIONS_PATH, n=5)

    for q_idx, q in enumerate(questions, start=1):
        query = q["question"]
        print("\n" + "=" * 80)
        print(f"Question {q_idx}: {query}")

        retrieved = retrieve(
            query=query,
            index=index,
            documents=documents,
            encoder=encoder,
            k=TOP_K_RETRIEVAL,
        )

        ret_docs = [doc for doc, _, _ in retrieved]
        ret_ids = [doc_id for _, doc_id, _ in retrieved]
        ret_scores = [score for _, _, score in retrieved]

        poison_scores = score_documents(
            query=query,
            documents=ret_docs,
            model=stage1_model,
            tokenizer=stage1_tokenizer,
        )

        survivors = [
            (doc, score, ret_score, doc_id)
            for doc, score, ret_score, doc_id
            in zip(ret_docs, poison_scores, ret_scores, ret_ids)
            if score <= 0.5
        ]

        if not survivors:
            print("Stage 1 filtered everything; keeping lowest poison-score doc.")
            min_idx = min(range(len(poison_scores)), key=lambda i: poison_scores[i])
            survivors = [
                (ret_docs[min_idx], poison_scores[min_idx], ret_scores[min_idx], ret_ids[min_idx])
            ]

        s1_docs, s1_scores, s1_ret_scores, s1_ids = zip(*survivors)
        s1_embs = np.array([doc_embeddings[doc_id] for doc_id in s1_ids])

        stage2_ranked = score_and_rerank_documents(
            documents=list(s1_docs),
            doc_embeddings=s1_embs,
            classifier_scores=list(s1_scores),
            retrieval_scores=list(s1_ret_scores),
            top_k=TOP_K_AFTER_TRUST,
        )

        print(f"Retrieved docs: {len(ret_docs)}")
        print(f"Stage 1 survivors: {len(s1_docs)}")
        print(f"Stage 2 returned: {len(stage2_ranked)} docs")

        assert len(stage2_ranked) <= TOP_K_AFTER_TRUST
        assert all("document" in item for item in stage2_ranked)
        assert all("trust_score" in item for item in stage2_ranked)

        print("\nStage 2 top docs:")
        for item in stage2_ranked:
            doc_preview = item["document"][:160].replace("\n", " ")
            print(f"Rank {item['rank']} | trust={item['trust_score']:.4f} | {doc_preview}...")

    print("\nStage 2 NQ test passed!")


if __name__ == "__main__":
    main()
