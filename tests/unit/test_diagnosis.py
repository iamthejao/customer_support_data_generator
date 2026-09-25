"""Unit tests for diagnosis beats and the speakers' views of the plan."""

from __future__ import annotations

from csfd.diagnosis import (
    CandidateCause,
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
        candidate_causes=[
            CandidateCause(
                cause=ROOT,
                is_root_cause=True,
                resolution_steps=["replace the door seal"],
                parts=["door seal"],
                verification="run the vacuum test",
                verification_finding="vacuum reaches 40 mbar",
                workaround="press with the door clamped",
            ),
            CandidateCause(cause="worn vacuum pump", resolution_steps=["replace the pump"]),
            CandidateCause(cause="loose hose", resolution_steps=["tighten the hose clamp"]),
        ],
        checks=[
            DiagnosticCheck(
                check=f"Check {i + 1}",
                how_to_check=f"do step {i + 1}",
                finding=f"finding {i + 1}",
                confirms_cause=i == checks - 1,
            )
            for i in range(checks)
        ],
    )


def test_single_contact_runs_every_check_and_reaches_the_cause() -> None:
    [beat] = plan_beats(_plan(), ["final"], ProblemState.FIXED_VERIFIED)
    assert beat.checks == [0, 1, 2]
    assert beat.cause_confirmed is True
    assert beat.problem_state == ProblemState.FIXED_VERIFIED
    assert beat.next_check is None


def test_follow_up_contact_agrees_the_next_check_as_the_customers_test() -> None:
    first, second = plan_beats(_plan(), ["follow_up", "final"], ProblemState.PENDING_PART)
    assert first.checks == [0]
    assert first.next_check == 1
    assert first.problem_state == ProblemState.PENDING_CUSTOMER_TEST
    assert first.cause_confirmed is False
    assert second.checks == [1, 2]  # the final contact takes the remainder
    assert second.cause_confirmed is True
    assert second.problem_state == ProblemState.PENDING_PART


def test_final_contact_keeps_a_check_when_there_are_fewer_checks_than_contacts() -> None:
    first, second = plan_beats(_plan(1), ["dropped", "final"], ProblemState.FIXED_VERIFIED)
    assert (first.checks, second.checks) == ([], [0])


def test_dropped_contact_runs_no_checks_and_the_next_contacts_take_them() -> None:
    first, second = plan_beats(_plan(), ["dropped", "final"], ProblemState.FIXED_VERIFIED)
    assert (first.checks, second.checks) == ([], [0, 1, 2])
    beats = plan_beats(_plan(), ["follow_up", "dropped", "final"], ProblemState.FIXED_VERIFIED)
    assert [b.checks for b in beats] == [[0], [], [1, 2]]
    # The dropped contact reports no result, so the follow-up agrees no specific check.
    assert beats[0].next_check is None


def test_follow_up_agrees_only_the_next_contacts_first_check() -> None:
    beats = plan_beats(_plan(1), ["follow_up", "follow_up", "final"], ProblemState.FIXED_VERIFIED)
    assert [b.checks for b in beats] == [[], [], [0]]
    assert [b.next_check for b in beats] == [None, 0, None]


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


def test_agent_guide_lists_every_cause_with_its_fix_but_not_the_truth() -> None:
    plan = _plan()
    early = ContactBeat(
        checks=[0, 1], next_check=2, problem_state=ProblemState.PENDING_CUSTOMER_TEST
    )
    guide = agent_guide(plan, early, done_checks=[], seed_label="x")
    assert "finding" not in repr(guide["checks"])
    assert "is_root_cause" not in repr(guide)
    fixes = {c["cause"]: c["resolution_steps"] for c in guide["candidate_causes"]}
    assert fixes == {
        ROOT: ["replace the door seal"],
        "worn vacuum pump": ["replace the pump"],
        "loose hose": ["tighten the hose clamp"],
    }
    assert guide["beat_steps"] == [1, 2] and guide["next_step"] == 3

    # The contact that reaches the cause gets the same guide, only its steps differ.
    final = ContactBeat(checks=[2], cause_confirmed=True, problem_state=ProblemState.FIXED_VERIFIED)
    reached = agent_guide(plan, final, done_checks=[0, 1], seed_label="x")
    assert reached["candidate_causes"] == guide["candidate_causes"]
    assert reached["done_steps"] == [1, 2]


def test_candidate_cause_order_is_seeded_not_the_plans() -> None:
    plan = _plan()
    beat = ContactBeat(checks=[0])
    orders = {
        tuple(
            c["cause"]
            for c in agent_guide(plan, beat, done_checks=[], seed_label=f"p{i}")["candidate_causes"]
        )
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
