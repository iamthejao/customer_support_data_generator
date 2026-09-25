"""Unit tests for diagnosis beats and the speakers' views of the plan."""

from __future__ import annotations

from csfd.diagnosis import (
    ContactBeat,
    DiagnosisPlan,
    DiagnosticCheck,
    agent_guide,
    customer_findings,
    plan_beats,
)
from csfd.outcomes import ProblemState

ROOT = "cracked door seal"


def _plan(checks: int = 3) -> DiagnosisPlan:
    return DiagnosisPlan(
        candidate_causes=[ROOT, "worn vacuum pump", "loose hose"],
        checks=[
            DiagnosticCheck(
                check=f"Check {i + 1}",
                how_to_check=f"do step {i + 1}",
                finding=f"finding {i + 1}",
                confirms_cause=i == checks - 1,
            )
            for i in range(checks)
        ],
        resolution_steps=["replace the door seal"],
        verification="run the vacuum test",
        verification_finding="vacuum reaches 40 mbar",
        workaround="press with the door clamped",
        parts=["door seal"],
    )


def test_single_contact_runs_every_check_and_reaches_the_cause() -> None:
    [beat] = plan_beats(_plan(), ["final"], ProblemState.FIXED_VERIFIED)
    assert beat.checks == [0, 1, 2]
    assert beat.cause_confirmed is True
    assert beat.problem_state == ProblemState.FIXED_VERIFIED
    assert beat.next_check is None


def test_follow_up_contact_agrees_the_next_check_as_the_customers_test() -> None:
    first, second = plan_beats(_plan(), ["follow_up", "final"], ProblemState.PENDING_PART)
    assert first.checks == [0, 1]
    assert first.next_check == 2
    assert first.problem_state == ProblemState.PENDING_CUSTOMER_TEST
    assert first.cause_confirmed is False
    assert second.checks == [2]
    assert second.cause_confirmed is True
    assert second.problem_state == ProblemState.PENDING_PART


def test_dropped_contact_leaves_no_state_and_agrees_nothing() -> None:
    first, _ = plan_beats(_plan(), ["dropped", "final"], ProblemState.FIXED_VERIFIED)
    assert first.problem_state is None
    assert first.next_check is None


def test_states_that_stop_short_never_run_the_confirming_check() -> None:
    for state in (
        ProblemState.ESCALATED_OPEN,
        ProblemState.ABANDONED,
        ProblemState.PENDING_CUSTOMER_TEST,
    ):
        [beat] = plan_beats(_plan(), ["final"], state)
        assert beat.checks == [0, 1]
        assert beat.cause_confirmed is False
    [beat] = plan_beats(_plan(), ["final"], ProblemState.PENDING_CUSTOMER_TEST)
    assert beat.next_check == 2


def test_no_plan_still_carries_the_planned_state() -> None:
    beats = plan_beats(None, ["follow_up", "final"], ProblemState.WORKAROUND)
    assert [b.checks for b in beats] == [[], []]
    assert [b.problem_state for b in beats] == [
        ProblemState.PENDING_CUSTOMER_TEST,
        ProblemState.WORKAROUND,
    ]


def test_agent_guide_never_carries_the_findings_or_an_unreached_fix() -> None:
    plan = _plan()
    early = ContactBeat(
        checks=[0, 1], next_check=2, problem_state=ProblemState.PENDING_CUSTOMER_TEST
    )
    guide = agent_guide(plan, early, done_checks=[], seed_label="x")
    text = repr(guide)
    assert "finding" not in text
    assert "replace the door seal" not in text
    assert sorted(guide["candidate_causes"]) == sorted(plan.candidate_causes)
    assert guide["beat_steps"] == [1, 2] and guide["next_step"] == 3

    final = ContactBeat(checks=[2], cause_confirmed=True, problem_state=ProblemState.FIXED_VERIFIED)
    guide = agent_guide(plan, final, done_checks=[0, 1], seed_label="x")
    assert guide["resolution_steps"] == ["replace the door seal"]
    assert guide["verification"] == "run the vacuum test"
    assert guide["done_steps"] == [1, 2]


def test_candidate_cause_order_is_seeded_not_the_plans() -> None:
    plan = _plan()
    beat = ContactBeat(checks=[0])
    orders = {
        tuple(agent_guide(plan, beat, done_checks=[], seed_label=f"p{i}")["candidate_causes"])
        for i in range(12)
    }
    assert len(orders) > 1


def test_customer_sees_findings_up_to_this_contact_only() -> None:
    plan = _plan()
    beat = ContactBeat(checks=[1], next_check=2, problem_state=ProblemState.PENDING_CUSTOMER_TEST)
    view = customer_findings(plan, beat, done_checks=[0])
    assert [f["finding"] for f in view["findings"]] == ["finding 1", "finding 2"]
    assert ROOT not in repr(view)
    assert "after_fix" not in view
    fixed = ContactBeat(checks=[2], cause_confirmed=True, problem_state=ProblemState.FIXED_VERIFIED)
    assert (
        customer_findings(plan, fixed, done_checks=[0, 1])["after_fix"] == "vacuum reaches 40 mbar"
    )
