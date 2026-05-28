"""Perplexity-based filtering baseline using GPT-2.

Zou et al. report >87% ASR even against this filter, establishing a weak
baseline. High perplexity = suspicious document.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_GPT2_MODEL = "openai-community/gpt2"


def load_perplexity_model(model_name: str = _GPT2_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def compute_perplexity(text: str, model, tokenizer, max_length: int = 512) -> float:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return float(torch.exp(outputs.loss).item())


def filter_by_perplexity(
    documents: list[str],
    model,
    tokenizer,
    threshold: float = 200.0,
) -> list[tuple[str, float]]:
    """Remove documents with perplexity above threshold. Returns (doc, ppl) pairs."""
    scored = [(doc, compute_perplexity(doc, model, tokenizer)) for doc in documents]
    survivors = [(doc, ppl) for doc, ppl in scored if ppl <= threshold]
    if not survivors:
        survivors = [min(scored, key=lambda x: x[1])]
    return survivors
