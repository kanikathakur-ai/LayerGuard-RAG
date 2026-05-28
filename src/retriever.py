import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import EMBEDDING_MODEL, EMBEDDING_DIM, TOP_K_RETRIEVAL


def _load_encoder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    # IndexFlatIP requires L2-normalized vectors to compute cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-10)


def build_index(documents: list[str], encoder: SentenceTransformer | None = None) -> tuple[faiss.Index, np.ndarray]:
    """Encode documents and build a FAISS IndexFlatIP. Returns (index, embeddings)."""
    if encoder is None:
        encoder = _load_encoder()
    print(f"Encoding {len(documents)} documents...")
    embeddings = encoder.encode(documents, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    embeddings = _normalize(embeddings).astype("float32")

    if len(documents) > 100_000:
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, 256, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    index.add(embeddings)
    return index, embeddings


def save_index(index: faiss.Index, embeddings: np.ndarray, index_path: str, embeddings_path: str) -> None:
    faiss.write_index(index, index_path)
    np.save(embeddings_path, embeddings)


def load_index(index_path: str, embeddings_path: str) -> tuple[faiss.Index, np.ndarray]:
    index = faiss.read_index(index_path)
    embeddings = np.load(embeddings_path)
    return index, embeddings


def retrieve(
    query: str,
    index: faiss.Index,
    documents: list[str],
    encoder: SentenceTransformer,
    k: int = TOP_K_RETRIEVAL,
) -> list[tuple[str, int, float]]:
    """Return top-k (doc_text, doc_id, score) tuples for a query."""
    q_emb = encoder.encode([query], convert_to_numpy=True)
    q_emb = _normalize(q_emb).astype("float32")
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append((documents[idx], int(idx), float(score)))
    return results


def inject_documents(
    index: faiss.Index,
    existing_embeddings: np.ndarray,
    existing_docs: list[str],
    new_docs: list[str],
    encoder: SentenceTransformer,
) -> tuple[faiss.Index, np.ndarray, list[str]]:
    """Add new documents to the index. Returns updated (index, embeddings, documents)."""
    new_embs = encoder.encode(new_docs, convert_to_numpy=True)
    new_embs = _normalize(new_embs).astype("float32")
    index.add(new_embs)
    updated_embeddings = np.vstack([existing_embeddings, new_embs])
    updated_docs = existing_docs + new_docs
    return index, updated_embeddings, updated_docs


def load_documents(docs_path: str) -> list[str]:
    """Load documents from a JSONL file. Expects {"text": "..."} per line."""
    docs = []
    with open(docs_path) as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj["text"])
    return docs
