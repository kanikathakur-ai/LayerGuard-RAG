"""Stage 1: DeBERTa-v3-base binary classifier.

Filters documents likely to be poisoned before they reach the LLM.
Implement fully in Week 1.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import STAGE1_MAX_LENGTH, STAGE1_MODEL, STAGE1_THRESHOLD


def load_classifier(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def score_documents(
    query: str,
    documents: list[str],
    model,
    tokenizer,
    batch_size: int = 16,
) -> list[float]:
    """Return P(poisoned) for each document."""
    scores = []
    device = next(model.parameters()).device
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        enc = tokenizer(
            [query] * len(batch_docs),
            batch_docs,
            truncation=True,
            max_length=STAGE1_MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()  # P(poisoned)
        scores.extend(probs)
    return scores


def filter_documents(
    query: str,
    documents: list[str],
    model,
    tokenizer,
    threshold: float = STAGE1_THRESHOLD,
) -> tuple[list[str], list[float]]:
    """Return (surviving_docs, their_scores) after removing P(poisoned) > threshold."""
    scores = score_documents(query, documents, model, tokenizer)
    survivors = [(doc, s) for doc, s in zip(documents, scores) if s <= threshold]
    if not survivors:
        # Safety: if everything filtered, keep lowest-scoring doc
        min_idx = min(range(len(scores)), key=lambda i: scores[i])
        survivors = [(documents[min_idx], scores[min_idx])]
    docs, kept_scores = zip(*survivors)
    return list(docs), list(kept_scores)


def train_classifier(train_data, val_data, output_dir: str):
    """Fine-tune DeBERTa-v3-base. Implemented in scripts/train_stage1.py."""
    raise NotImplementedError("See scripts/train_stage1.py")
