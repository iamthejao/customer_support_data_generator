"""Tests for the deterministic allocation planner."""

from __future__ import annotations

import pytest

from csfd.allocator import (
    ProblemRef,
    build_allocation_plan,
)
from csfd.ticket_types.definitions import ProblemComplexity, TicketType

TYPE_PROPS = {
    "docs_request": 0.50,
    "l1": 0.30,
    "l2": 0.10,
    "l3": 0.10,
}
TIER_PROPS = {"standard": 0.6, "premium": 0.3, "enterprise": 0.1}
TONE_PROPS = {
    "docs_request": {"neutral": 0.7, "polite": 0.3},
    "l1": {"neutral": 0.5, "frustrated": 0.3, "polite": 0.2},
    "l2": {"frustrated": 0.5, "urgent": 0.3, "neutral": 0.2},
    "l3": {"urgent": 0.4, "frustrated": 0.4, "neutral": 0.2},
}


def _problems(by_complexity: dict[ProblemComplexity, int]) -> list[ProblemRef]:
    out: list[ProblemRef] = []
    n = 0
    for c, k in by_complexity.items():
        for _ in range(k):
            n += 1
            out.append(ProblemRef(id=f"p{n:03d}", complexity=c))
    return out


def test_plan_total_matches_target() -> None:
    plan = build_allocation_plan(
        total=100,
        type_proportions=TYPE_PROPS,
        tier_proportions=TIER_PROPS,
        tone_proportions_per_type=TONE_PROPS,
        problems=_problems(
            {ProblemComplexity.SIMPLE: 5, ProblemComplexity.MEDIUM: 5, ProblemComplexity.COMPLEX: 5}
        ),
        assignment_strategy="complexity_weighted",
    )
    assert len(plan.slots) == 100


def test_plan_type_counts_exact() -> None:
    plan = build_allocation_plan(
        total=100,
        type_proportions=TYPE_PROPS,
        tier_proportions=TIER_PROPS,
        tone_proportions_per_type=TONE_PROPS,
        problems=_problems(
            {ProblemComplexity.SIMPLE: 5, ProblemComplexity.MEDIUM: 5, ProblemComplexity.COMPLEX: 5}
        ),
        assignment_strategy="complexity_weighted",
    )
    assert plan.counts_by_type == {
        TicketType.DOCS_REQUEST: 50,
        TicketType.L1: 30,
        TicketType.L2: 10,
        TicketType.L3: 10,
    }
    counts: dict[TicketType, int] = {}
    for s in plan.slots:
        counts[s.ticket_type] = counts.get(s.ticket_type, 0) + 1
    assert counts == plan.counts_by_type


def test_plan_tier_counts_exact() -> None:
    plan = build_allocation_plan(
        total=100,
        type_proportions=TYPE_PROPS,
        tier_proportions=TIER_PROPS,
        tone_proportions_per_type=TONE_PROPS,
        problems=_problems(
            {ProblemComplexity.SIMPLE: 5, ProblemComplexity.MEDIUM: 5, ProblemComplexity.COMPLEX: 5}
        ),
        assignment_strategy="complexity_weighted",
    )
    tiers: dict[str, int] = {}
    for s in plan.slots:
        tiers[s.tier] = tiers.get(s.tier, 0) + 1
    assert tiers == {"standard": 60, "premium": 30, "enterprise": 10}


def test_plan_tone_per_type_exact() -> None:
    plan = build_allocation_plan(
        total=100,
        type_proportions=TYPE_PROPS,
        tier_proportions=TIER_PROPS,
        tone_proportions_per_type=TONE_PROPS,
        problems=_problems(
            {ProblemComplexity.SIMPLE: 5, ProblemComplexity.MEDIUM: 5, ProblemComplexity.COMPLEX: 5}
        ),
        assignment_strategy="complexity_weighted",
    )
    per: dict[TicketType, dict[str, int]] = {}
    for s in plan.slots:
        per.setdefault(s.ticket_type, {})
        per[s.ticket_type][s.tone] = per[s.ticket_type].get(s.tone, 0) + 1
    # docs_request: 50 tickets, 70/30 -> 35/15
    assert per[TicketType.DOCS_REQUEST] == {"neutral": 35, "polite": 15}
    # l1: 30 tickets, 50/30/20 -> 15/9/6
    assert per[TicketType.L1] == {"neutral": 15, "frustrated": 9, "polite": 6}
    # l2: 10 tickets, 50/30/20 -> 5/3/2
    assert per[TicketType.L2] == {"frustrated": 5, "urgent": 3, "neutral": 2}


def test_plan_deterministic_across_calls() -> None:
    problems = _problems(
        {
            ProblemComplexity.SIMPLE: 5,
            ProblemComplexity.MEDIUM: 5,
            ProblemComplexity.COMPLEX: 5,
        }
    )
    a = build_allocation_plan(
        total=100,
        type_proportions=TYPE_PROPS,
        tier_proportions=TIER_PROPS,
        tone_proportions_per_type=TONE_PROPS,
        problems=problems,
        assignment_strategy="complexity_weighted",
    )
    b = build_allocation_plan(
        total=100,
        type_proportions=TYPE_PROPS,
        tier_proportions=TIER_PROPS,
        tone_proportions_per_type=TONE_PROPS,
        problems=problems,
        assignment_strategy="complexity_weighted",
    )
    assert a.slots == b.slots


def test_complexity_weighted_prefers_complex_for_l3() -> None:
    problems = _problems({ProblemComplexity.SIMPLE: 3, ProblemComplexity.COMPLEX: 3})
    plan = build_allocation_plan(
        total=10,
        type_proportions={"docs_request": 0.0, "l1": 0.0, "l2": 0.0, "l3": 1.0},
        tier_proportions={"standard": 1.0},
        tone_proportions_per_type={
            "docs_request": {"neutral": 1.0},
            "l1": {"neutral": 1.0},
            "l2": {"neutral": 1.0},
            "l3": {"neutral": 1.0},
        },
        problems=problems,
        assignment_strategy="complexity_weighted",
    )
    l3_problem_ids = [s.problem_id for s in plan.slots if s.ticket_type == TicketType.L3]
    complex_ids = {p.id for p in problems if p.complexity == ProblemComplexity.COMPLEX}
    # Should be drawn exclusively from complex bucket (preference exhausts before fallback).
    assert set(l3_problem_ids).issubset(complex_ids)


def test_complexity_weighted_prefers_simple_for_docs() -> None:
    problems = _problems({ProblemComplexity.SIMPLE: 3, ProblemComplexity.COMPLEX: 3})
    plan = build_allocation_plan(
        total=10,
        type_proportions={"docs_request": 1.0, "l1": 0.0, "l2": 0.0, "l3": 0.0},
        tier_proportions={"standard": 1.0},
        tone_proportions_per_type={
            "docs_request": {"neutral": 1.0},
            "l1": {"neutral": 1.0},
            "l2": {"neutral": 1.0},
            "l3": {"neutral": 1.0},
        },
        problems=problems,
        assignment_strategy="complexity_weighted",
    )
    docs_ids = [s.problem_id for s in plan.slots if s.ticket_type == TicketType.DOCS_REQUEST]
    simple_ids = {p.id for p in problems if p.complexity == ProblemComplexity.SIMPLE}
    assert set(docs_ids).issubset(simple_ids)


def test_uniform_changes_with_seed() -> None:
    problems = _problems({ProblemComplexity.SIMPLE: 5, ProblemComplexity.COMPLEX: 5})
    a = build_allocation_plan(
        total=10,
        type_proportions={"docs_request": 1.0, "l1": 0.0, "l2": 0.0, "l3": 0.0},
        tier_proportions={"standard": 1.0},
        tone_proportions_per_type={
            "docs_request": {"neutral": 1.0},
            "l1": {"neutral": 1.0},
            "l2": {"neutral": 1.0},
            "l3": {"neutral": 1.0},
        },
        problems=problems,
        assignment_strategy="uniform",
        seed=1,
    )
    b = build_allocation_plan(
        total=10,
        type_proportions={"docs_request": 1.0, "l1": 0.0, "l2": 0.0, "l3": 0.0},
        tier_proportions={"standard": 1.0},
        tone_proportions_per_type={
            "docs_request": {"neutral": 1.0},
            "l1": {"neutral": 1.0},
            "l2": {"neutral": 1.0},
            "l3": {"neutral": 1.0},
        },
        problems=problems,
        assignment_strategy="uniform",
        seed=2,
    )
    # Same seed deterministic, different seed produces different ordering.
    assert [s.problem_id for s in a.slots] != [s.problem_id for s in b.slots]


def test_total_zero_yields_empty_plan() -> None:
    plan = build_allocation_plan(
        total=0,
        type_proportions=TYPE_PROPS,
        tier_proportions=TIER_PROPS,
        tone_proportions_per_type=TONE_PROPS,
        problems=_problems({ProblemComplexity.SIMPLE: 1}),
        assignment_strategy="complexity_weighted",
    )
    assert plan.slots == ()


def test_no_problems_with_positive_total_raises() -> None:
    with pytest.raises(ValueError):
        build_allocation_plan(
            total=10,
            type_proportions=TYPE_PROPS,
            tier_proportions=TIER_PROPS,
            tone_proportions_per_type=TONE_PROPS,
            problems=[],
            assignment_strategy="complexity_weighted",
        )


def test_missing_tone_config_raises() -> None:
    with pytest.raises(ValueError):
        build_allocation_plan(
            total=10,
            type_proportions={"docs_request": 1.0, "l1": 0.0, "l2": 0.0, "l3": 0.0},
            tier_proportions={"standard": 1.0},
            tone_proportions_per_type={"l1": {"neutral": 1.0}},  # docs_request missing
            problems=_problems({ProblemComplexity.SIMPLE: 1}),
            assignment_strategy="complexity_weighted",
        )
