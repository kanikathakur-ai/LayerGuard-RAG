"""Load pre-generated PoisonedRAG adversarial passages as LayerGuard PoisonedExamples.

Reuses the PoisonedExample dataclass from inject_poison.py so this is a drop-in
replacement for run_attack() in both eval harnesses.

The LM_targeted construction from the official PoisonedRAG repo (src/attack.py:92-93)
prepends the question to each adv_text to guarantee retrieval similarity without needing
retriever access (black-box). This is the only variant we support — the white-box hotflip
variant is Contriever-specific and won't transfer to LayerGuard's MiniLM retriever.
"""

import json
from typing import Optional

from src.attacks.inject_poison import PoisonedExample
from src.retriever import retrieve


def load_poisonedrag(
    path: str,
    n_docs: int = 5,
    prepend_question: bool = True,
) -> list[PoisonedExample]:
    """Parse adv_targeted_results/{dataset}.json into PoisonedExamples.

    Each entry's `incorrect answer` becomes target_answer; `adv_texts` become
    the poison docs. prepend_question=True applies the LM_targeted construction:
    `question + ". " + adv_text`, which is what makes them retrieve for the query.
    """
    with open(path) as f:
        data = json.load(f)

    examples = []
    for entry in data.values():
        question = entry["question"]
        target_answer = entry["incorrect answer"]
        adv_texts = entry["adv_texts"][:n_docs]
        if prepend_question:
            poison_docs = [f"{question}. {t}" for t in adv_texts]
        else:
            poison_docs = list(adv_texts)
        examples.append(PoisonedExample(
            question=question,
            target_answer=target_answer,
            poisoned_docs=poison_docs,
        ))
    return examples


def poisonedrag_questions(path: str) -> list[dict]:
    """Emit harness-compatible target question dicts from adv_targeted_results.

    Schema matches data/test_questions.jsonl:
      {question, answer, target_answer, gold_doc_id}
    gold_doc_id is None; call resolve_gold_doc_ids() to fill it in.
    """
    with open(path) as f:
        data = json.load(f)

    questions = []
    for entry in data.values():
        questions.append({
            "question": entry["question"],
            "answer": entry["correct answer"],
            "target_answer": entry["incorrect answer"],
            "gold_doc_id": None,
        })
    return questions


def resolve_gold_doc_ids(
    questions: list[dict],
    index,
    documents: list[str],
    encoder,
    top_k: int = 20,
) -> None:
    """Fill gold_doc_id in-place for questions whose correct answer appears in top-k.

    Retrieves top_k docs for each question and sets gold_doc_id to the first doc
    whose text contains the normalized correct answer as a substring. Leaves None
    if no match is found (answer not in the 50k corpus subset).
    """
    for q in questions:
        if q.get("gold_doc_id") is not None:
            continue
        answer_norm = q["answer"].strip().lower()
        retrieved = retrieve(q["question"], index, documents, encoder, k=top_k)
        for _doc_text, doc_id, _score in retrieved:
            if answer_norm in _doc_text.lower():
                q["gold_doc_id"] = doc_id
                break
