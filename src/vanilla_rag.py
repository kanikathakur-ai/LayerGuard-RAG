"""End-to-end vanilla RAG baseline — no defense."""

import time
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TOP_K_RETRIEVAL
from src.retriever import retrieve
from src.generator import generate_answer


def run_vanilla_rag(
    query: str,
    index,
    documents: list[str],
    encoder: SentenceTransformer,
    model,
    tokenizer,
    k: int = TOP_K_RETRIEVAL,
) -> dict:
    """Retrieve top-k docs and generate an answer with no poisoning defense.

    Returns a dict with answer, retrieved docs, scores, and wall-clock timing.
    """
    t0 = time.perf_counter()
    retrieved = retrieve(query, index, documents, encoder, k=k)
    t_retrieve = time.perf_counter() - t0

    context_docs = [doc for doc, _, _ in retrieved]
    doc_scores = [(doc_id, score) for _, doc_id, score in retrieved]

    t1 = time.perf_counter()
    answer = generate_answer(query, context_docs, model, tokenizer)
    t_generate = time.perf_counter() - t1

    return {
        "answer": answer,
        "retrieved_docs": context_docs,
        "doc_scores": doc_scores,
        "timings": {
            "retrieve_s": t_retrieve,
            "generate_s": t_generate,
            "total_s": t_retrieve + t_generate,
        },
    }


def batch_vanilla_rag(
    queries: list[str],
    index,
    documents: list[str],
    encoder: SentenceTransformer,
    model,
    tokenizer,
    k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    from tqdm import tqdm
    return [
        run_vanilla_rag(q, index, documents, encoder, model, tokenizer, k=k)
        for q in tqdm(queries, desc="Vanilla RAG")
    ]
