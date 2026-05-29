"""Device selection with priority cuda > mps > cpu.

Used so the same Stage 3 code runs on the remote CUDA box and on a Mac (MPS)
for quick local smoke tests, falling back to CPU anywhere else.
"""

from __future__ import annotations

import torch


def get_device() -> str:
    """Return the best available device string: 'cuda' > 'mps' > 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def hf_pipeline_device(device: str | None = None):
    """Map a device string to the HuggingFace ``pipeline(device=...)`` argument.

    HF pipelines expect an int CUDA ordinal, the string ``"mps"``, or ``-1`` for
    CPU. If ``device`` is None, the best available device is selected.
    """
    if device is None:
        device = get_device()
    if device == "cuda":
        return 0
    if device == "mps":
        return "mps"
    return -1
