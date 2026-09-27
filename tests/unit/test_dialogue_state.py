"""Unit tests for dialogue schemas, resolved-flag derivation, and transcript assembly."""

from __future__ import annotations

import pytest

from csfd.outcomes import Commitment, ContactEnding, ProblemState
from csfd.pipeline import (
    DialogueTurnOutput,
    IncomingRequestOutput,
    ResolutionOutput,
    _assemble_resolution,
)


def test_dialogue_turn_output_fields() -> None:
    t = DialogueTurnOutput(speaker="customer", content="hi", done=False)
    assert t.speaker == "customer"
    assert t.done is False
    assert t.done_reason is None


def test_incoming_request_output_fields() -> None:
    ir = IncomingRequestOutput(subject="S", body="B")
    assert ir.subject == "S"
    assert ir.body == "B"


@pytest.mark.parametrize(
    ("end_reason", "last_reason", "expected"),
    [
        ("agent_done", "resolved", ContactEnding.CUSTOMER_SATISFIED),
        ("customer_done", "customer_satisfied", ContactEnding.CUSTOMER_SATISFIED),
        ("agent_done", "follow_up", ContactEnding.AGREED_NEXT_STEP),
        ("agent_done", "escalation", ContactEnding.AGREED_NEXT_STEP),
        ("customer_done", "customer_frustrated", ContactEnding.CUSTOMER_FRUSTRATED),
        ("agent_done", None, ContactEnding.AGREED_NEXT_STEP),
        ("cap_hit", "resolved", ContactEnding.CAP_HIT),
        ("dropped", None, ContactEnding.DROPPED),
    ],
)
def test_assembled_contact_ending(
    end_reason: str, last_reason: str | None, expected: ContactEnding
) -> None:
    turns = [DialogueTurnOutput(speaker="agent", content="x", done=True, done_reason=last_reason)]
    res = _assemble_resolution(subject="S", body="x", turns=turns, end_reason=end_reason)  # type: ignore[arg-type]
    assert res.contact_ending == expected


def test_assemble_resolution_builds_output() -> None:
    turns = [
        DialogueTurnOutput(speaker="customer", content="opening", done=False),
        DialogueTurnOutput(
            speaker="agent",
            content="fixed it",
            done=True,
            done_reason="resolved",
            commitments=[Commitment(who="agent", what="send the report", due="today")],
        ),
    ]
    res = _assemble_resolution(
        subject="Sub",
        body="opening",
        turns=turns,
        end_reason="agent_done",
        problem_state=ProblemState.FIXED_VERIFIED,
    )
    assert isinstance(res, ResolutionOutput)
    assert res.subject == "Sub"
    assert res.body == "opening"
    assert res.turns == turns
    assert res.end_reason == "agent_done"
    assert res.resolved is True
    assert res.commitments == [Commitment(who="agent", what="send the report", due="today")]


@pytest.mark.parametrize(
    ("state", "resolved"),
    [
        (ProblemState.FIXED_VERIFIED, True),
        (ProblemState.FIXED_UNVERIFIED, True),
        (ProblemState.PENDING_VISIT, False),
        (ProblemState.WORKAROUND, False),
        (None, False),
    ],
)
def test_resolved_is_derived_from_the_problem_state(
    state: ProblemState | None, resolved: bool
) -> None:
    # "We'll send a technician" is an agreed plan, not a fix.
    turns = [DialogueTurnOutput(speaker="agent", content="x", done=True, done_reason="resolved")]
    res = _assemble_resolution(
        subject="S", body="x", turns=turns, end_reason="agent_done", problem_state=state
    )
    assert res.resolved is resolved
