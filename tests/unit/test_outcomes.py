"""Unit tests for planned problem states and contact endings."""

from __future__ import annotations

from collections import Counter

import pytest

from csfd.outcomes import (
    ContactEnding,
    ProblemState,
    is_resolved,
    plan_problem_states,
    planned_ending,
)


def test_only_fixed_states_count_as_resolved() -> None:
    assert {s for s in ProblemState if is_resolved(s)} == {
        ProblemState.FIXED_VERIFIED,
        ProblemState.FIXED_UNVERIFIED,
    }
    assert is_resolved(None) is False
    assert is_resolved("pending_visit") is False


@pytest.mark.parametrize(
    ("state", "end_mode", "ending"),
    [
        (None, "dropped", ContactEnding.DROPPED),
        (ProblemState.PENDING_CUSTOMER_TEST, "follow_up", ContactEnding.AGREED_NEXT_STEP),
        (ProblemState.FIXED_VERIFIED, "final", ContactEnding.CUSTOMER_SATISFIED),
        (ProblemState.NOT_A_FAULT, "final", ContactEnding.CUSTOMER_SATISFIED),
        (ProblemState.PENDING_VISIT, "final", ContactEnding.AGREED_NEXT_STEP),
        (ProblemState.ESCALATED_OPEN, "final", ContactEnding.AGREED_NEXT_STEP),
        (ProblemState.ABANDONED, "final", ContactEnding.CUSTOMER_FRUSTRATED),
    ],
)
def test_planned_ending(state: ProblemState | None, end_mode: str, ending: ContactEnding) -> None:
    assert planned_ending(state, end_mode) == ending


def test_states_match_proportions_exactly_when_every_problem_allows_them() -> None:
    slots = list(range(1, 21))
    states = plan_problem_states(
        slots, {}, {"fixed_verified": 0.5, "pending_visit": 0.25, "abandoned": 0.25}, seed=3
    )
    assert Counter(states.values()) == {
        ProblemState.FIXED_VERIFIED: 10,
        ProblemState.PENDING_VISIT: 5,
        ProblemState.ABANDONED: 5,
    }
    assert states == plan_problem_states(
        slots, {}, {"fixed_verified": 0.5, "pending_visit": 0.25, "abandoned": 0.25}, seed=3
    )


def test_states_stay_within_each_problems_viable_outcomes() -> None:
    slots = list(range(1, 11))
    viable = {i: (["not_a_fault", "fixed_verified"] if i % 2 else ["pending_part"]) for i in slots}
    states = plan_problem_states(
        slots, viable, {"fixed_verified": 0.5, "pending_part": 0.5}, seed=1
    )
    for index, state in states.items():
        assert state.value in viable[index]
    # Odd slots cannot take pending_part, so they take the other weighted state.
    assert {states[i] for i in slots if i % 2} == {ProblemState.FIXED_VERIFIED}


def test_unweighted_viable_outcomes_fall_back_to_the_first() -> None:
    states = plan_problem_states(
        [1], {1: ["not_a_fault", "abandoned"]}, {"fixed_verified": 1.0}, seed=0
    )
    assert states == {1: ProblemState.NOT_A_FAULT}
