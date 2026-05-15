"""Seeded RNG helpers for reproducible sampling."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from random import Random


def derive_rng(seed: int, label: str) -> Random:
    """Derive a deterministic Random instance for a (seed, label) pair.

    Different labels under the same seed produce independent sub-streams.
    """
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    sub_seed = int.from_bytes(digest[:8], "big", signed=False)
    return Random(sub_seed)


def weighted_choice(weights: Mapping[str, float], rng: Random) -> str:
    """Pick a key proportional to its weight. Raises if weights is empty."""
    if not weights:
        raise ValueError("weights must be non-empty")
    keys = list(weights.keys())
    values = [weights[k] for k in keys]
    return rng.choices(keys, weights=values, k=1)[0]
