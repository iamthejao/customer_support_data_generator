"""Deterministic, seeded samplers for Phase 2 ticket generation."""

from __future__ import annotations

from collections.abc import Mapping
from random import Random

from csfd.phases.phase1_kb.state import CommittedProblem
from csfd.ticket_types.definitions import TicketType
from csfd.utils.rng import weighted_choice


def sample_ticket_type(
    problem: CommittedProblem,
    rng: Random,
    *,
    weights_has_kb: Mapping[str, float],
    weights_no_kb: Mapping[str, float],
) -> TicketType:
    """Pick a ticket type for `problem`.

    Hard rule: when `problem.has_kb=False`, the result is always `L3` regardless
    of weights -- the no-KB path is the entire reason Phase 1 sets the flag.
    Otherwise pick proportionally to `weights_has_kb`.
    """
    if not problem.has_kb:
        picked = weighted_choice(weights_no_kb, rng)
        if picked != TicketType.L3.value:
            return TicketType.L3
        return TicketType(picked)
    return TicketType(weighted_choice(weights_has_kb, rng))


def sample_ticket_count(rng: Random, *, range_inclusive: tuple[int, int]) -> int:
    """Pick a uniform integer in [lo, hi] inclusive."""
    lo, hi = range_inclusive
    return rng.randint(lo, hi)
