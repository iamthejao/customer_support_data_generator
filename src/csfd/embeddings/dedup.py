"""Pure helpers for similarity-based dedup. No I/O."""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def find_duplicate(
    candidate: Sequence[float],
    existing: Sequence[tuple[str, Sequence[float]]],
    threshold: float,
) -> tuple[str, float] | None:
    """Return the highest-scoring (id, cosine) above threshold, else None.

    Ties keep the first-seen entry (stable wrt input order).
    """
    best: tuple[str, float] | None = None
    for ident, vec in existing:
        score = cosine(candidate, vec)
        if score >= threshold and (best is None or score > best[1]):
            best = (ident, score)
    return best
