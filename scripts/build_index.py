"""One-time script to encode the NQ corpus and build the FAISS index.

Usage:
    python scripts/build_index.py --docs data/nq/documents.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sentence_transformers import SentenceTransformer

from config import DOCS_PATH, EMBEDDING_MODEL, INDEX_PATH
from src.retriever import build_index, load_documents, save_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=DOCS_PATH, help="Path to documents.jsonl")
    parser.add_argument(
        "--index-out", default=INDEX_PATH, help="Output FAISS index path"
    )
    parser.add_argument(
        "--embeddings-out", default=INDEX_PATH.replace(".index", "_embeddings.npy")
    )
    args = parser.parse_args()

    print(f"Loading documents from {args.docs}...")
    documents = load_documents(args.docs)
    print(f"Loaded {len(documents)} documents.")

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    index, embeddings = build_index(documents, encoder)

    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    save_index(index, embeddings, args.index_out, args.embeddings_out)
    print(f"Saved index to {args.index_out}")
    print(f"Saved embeddings to {args.embeddings_out}")


if __name__ == "__main__":
    main()
