"""RAGuard ZKIP baseline stub.

ZKIP performs k+1 LLM forward passes per query (leave-one-out inference).
Reproduce from Kolhe et al. 2025 GitHub if feasible; otherwise compare
against their published numbers (ASR=0.000 on NQ, at high compute cost).
"""


def zkip_filter(query: str, documents: list[str], model, tokenizer) -> list[str]:
    """Leave-one-out inference filter.

    For each candidate document d_i, generate an answer without d_i.
    If the answer changes significantly, d_i is suspicious.

    This is a stub — implement from RAGuard paper/code in Week 3 if feasible.
    """
    raise NotImplementedError(
        "Implement from RAGuard (Kolhe et al. 2025) GitHub. "
        "Requires k+1 LLM forward passes per query."
    )
