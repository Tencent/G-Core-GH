# coding=utf-8
"""Deterministic filler-text helper (replacement for the defunct ``lipsum`` package)."""
from __future__ import annotations

import random
import string

# Wide charset + long tails → high token entropy for long-prompt smoke tests.
_ALPHABET = string.ascii_letters + string.digits
_MIN_WORD_LEN = 1
_MAX_WORD_LEN = 48


def generate_words(n: int, *, seed: int = 0) -> str:
    """Return ``n`` space-separated high-entropy pseudo-words."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rng = random.Random(seed)
    words = []
    for _ in range(n):
        length = rng.randint(_MIN_WORD_LEN, _MAX_WORD_LEN)
        words.append("".join(rng.choice(_ALPHABET) for _ in range(length)))
    return " ".join(words)
