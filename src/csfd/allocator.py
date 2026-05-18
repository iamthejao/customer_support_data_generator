"""Deterministic allocation planner for the case-generation pipeline.

Given a target total and proportions for ticket types, customer tiers, and
per-type tones, plus a list of generated problems with complexity tags, produce
a fully deterministic plan of (problem_id, ticket_type, tier, tone) slots.

The planner is the single point at which proportions are converted to integer
counts (via largest-remainder rounding) and at which problems are matched to
ticket slots (via a configurable strategy). Once the plan exists, every
downstream node in Phase 2 is deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from csfd.ticket_types.definitions import (
    TICKET_TYPE_METADATA,
    ProblemComplexity,
    TicketType,
)
from csfd.utils.rng import derive_rng, largest_remainder

AssignmentStrategy = Literal["uniform", "complexity_weighted"]


@dataclass(slots=True, frozen=True)
class ProblemRef:
    """Minimal problem identity needed by the allocator."""

    id: str
    complexity: ProblemComplexity


@dataclass(slots=True, frozen=True)
class TicketSlot:
    """One row of the allocation plan."""

    index: int  # 1-based sequential id across the run
    problem_id: str
    ticket_type: TicketType
    tier: str
    tone: str


@dataclass(slots=True, frozen=True)
class AllocationPlan:
    """Full deterministic plan for a Phase 2 run."""

    slots: tuple[TicketSlot, ...]
    counts_by_type: dict[TicketType, int]
    counts_by_tier: dict[str, int]
    counts_by_type_tone: dict[TicketType, dict[str, int]]


def build_allocation_plan(
    *,
    total: int,
    type_proportions: Mapping[str, float],
    tier_proportions: Mapping[str, float],
    tone_proportions_per_type: Mapping[str, Mapping[str, float]],
    problems: Sequence[ProblemRef],
    assignment_strategy: AssignmentStrategy,
    seed: int = 0,
) -> AllocationPlan:
    """Build a fully deterministic allocation plan."""
    if total < 0:
        raise ValueError("total must be >= 0")
    if total > 0 and not problems:
        raise ValueError("cannot allocate tickets with no problems")

    # 1) Per-type integer counts via largest-remainder.
    type_counts_raw = largest_remainder(type_proportions, total)
    type_counts: dict[TicketType, int] = {TicketType(k): v for k, v in type_counts_raw.items()}

    # 2) Per-tier integer counts via largest-remainder (over the global total).
    tier_counts = largest_remainder(tier_proportions, total)

    # 3) Per-(type, tone) integer counts via largest-remainder within each type's count.
    type_tone_counts: dict[TicketType, dict[str, int]] = {}
    for tt, n in type_counts.items():
        tone_props = tone_proportions_per_type.get(tt.value)
        if tone_props is None:
            raise ValueError(f"no tone_proportions configured for ticket type {tt.value}")
        type_tone_counts[tt] = largest_remainder(tone_props, n)

    # 4) Assign problems to ticket slots.
    assignments = _assign_problems(
        type_counts=type_counts,
        problems=problems,
        strategy=assignment_strategy,
        seed=seed,
    )

    # 5) Build per-type tone sequences (expanded from counts, in input order).
    type_tone_sequences: dict[TicketType, list[str]] = {}
    for tt, tone_count_map in type_tone_counts.items():
        seq: list[str] = []
        for tone, c in tone_count_map.items():
            seq.extend([tone] * c)
        type_tone_sequences[tt] = seq

    # 6) Build global tier sequence (expanded from counts, in input order).
    tier_sequence: list[str] = []
    for tier, c in tier_counts.items():
        tier_sequence.extend([tier] * c)

    # 7) Walk types in declared order; for each type, zip its problem assignments
    #    with its tone sequence; pull tier from the global tier sequence in order.
    slots: list[TicketSlot] = []
    tier_iter = iter(tier_sequence)
    index = 0
    for tt in type_counts:
        type_problem_ids = assignments[tt]
        tones = type_tone_sequences[tt]
        for problem_id, tone in zip(type_problem_ids, tones, strict=True):
            index += 1
            tier = next(tier_iter)
            slots.append(
                TicketSlot(
                    index=index,
                    problem_id=problem_id,
                    ticket_type=tt,
                    tier=tier,
                    tone=tone,
                )
            )

    return AllocationPlan(
        slots=tuple(slots),
        counts_by_type=type_counts,
        counts_by_tier=tier_counts,
        counts_by_type_tone=type_tone_counts,
    )


def _assign_problems(
    *,
    type_counts: Mapping[TicketType, int],
    problems: Sequence[ProblemRef],
    strategy: AssignmentStrategy,
    seed: int,
) -> dict[TicketType, list[str]]:
    """Return per-type lists of problem ids of the right length."""
    if strategy == "uniform":
        return _assign_uniform(type_counts=type_counts, problems=problems, seed=seed)
    if strategy == "complexity_weighted":
        return _assign_complexity_weighted(type_counts=type_counts, problems=problems)
    raise ValueError(f"unknown assignment_strategy: {strategy!r}")


def _assign_uniform(
    *,
    type_counts: Mapping[TicketType, int],
    problems: Sequence[ProblemRef],
    seed: int,
) -> dict[TicketType, list[str]]:
    """Round-robin all problems for each type. Seed shuffles the starting order
    so different seeds produce different but still deterministic plans."""
    out: dict[TicketType, list[str]] = {}
    rng = derive_rng(seed, "allocator:uniform")
    ordered_problems = sorted(problems, key=lambda p: p.id)
    base_order = [p.id for p in ordered_problems]
    # Deterministic per-run shuffle (driven by seed).
    rng.shuffle(base_order)

    for tt, n in type_counts.items():
        out[tt] = [base_order[i % len(base_order)] for i in range(n)] if base_order else []
    return out


def _assign_complexity_weighted(
    *,
    type_counts: Mapping[TicketType, int],
    problems: Sequence[ProblemRef],
) -> dict[TicketType, list[str]]:
    """Assign problems by matching each type's complexity preference.

    For each ticket type, walk its preference tuple in order; from the matching
    complexity bucket, round-robin problem ids. If preferred buckets are empty,
    fall back to other complexities (order: simple -> medium -> complex).
    """
    buckets: dict[ProblemComplexity, list[str]] = {
        ProblemComplexity.SIMPLE: [],
        ProblemComplexity.MEDIUM: [],
        ProblemComplexity.COMPLEX: [],
    }
    for p in sorted(problems, key=lambda x: x.id):
        buckets[p.complexity].append(p.id)

    fallback_order = (
        ProblemComplexity.SIMPLE,
        ProblemComplexity.MEDIUM,
        ProblemComplexity.COMPLEX,
    )

    out: dict[TicketType, list[str]] = {}
    for tt, n in type_counts.items():
        pref = TICKET_TYPE_METADATA[tt].complexity_preference
        ordered: list[str] = []
        seen: set[ProblemComplexity] = set()
        # Walk preference, then fallback; pick the FIRST non-empty bucket and
        # round-robin only within it. Only if that bucket is empty try the next.
        for c in (*pref, *fallback_order):
            if c in seen:
                continue
            seen.add(c)
            if buckets[c]:
                ordered = buckets[c]
                break
        if not ordered:
            out[tt] = []
            continue
        out[tt] = [ordered[i % len(ordered)] for i in range(n)]
    return out
