"""Seeded RNG helpers and deterministic quota allocation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from math import floor
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


def largest_remainder(proportions: Mapping[str, float], total: int) -> dict[str, int]:
    """Allocate `total` integer units across keys via Hamilton's largest-remainder method.

    Returns a dict of integer counts that sum exactly to `total`.
    Order of keys in the input mapping is the tiebreaker for equal remainders
    (earlier keys win), so the result is fully deterministic.

    Empty mapping returns an empty dict (total must be 0 in that case).
    """
    if total < 0:
        raise ValueError("total must be >= 0")
    if not proportions:
        if total != 0:
            raise ValueError("proportions is empty but total > 0")
        return {}

    keys = list(proportions.keys())
    weight_sum = sum(proportions[k] for k in keys)
    if weight_sum <= 0:
        raise ValueError("proportions must have a positive sum")

    raw = {k: proportions[k] / weight_sum * total for k in keys}
    floors = {k: floor(raw[k]) for k in keys}
    assigned = sum(floors.values())
    remainder = total - assigned

    # Distribute the remaining units to the keys with the largest fractional remainders.
    # Input-order is the tiebreaker (stable sort preserves it for equal remainders).
    ordered = sorted(
        keys,
        key=lambda k: (-(raw[k] - floors[k]), keys.index(k)),
    )
    out = dict(floors)
    for k in ordered[:remainder]:
        out[k] += 1
    return out
