"""Unit tests for dialogue schemas, resolved-flag derivation, and transcript assembly."""

from __future__ import annotations

import pytest

from csfd.pipeline import (
    DialogueTurnOutput,
    IncomingRequestOutput,
    ResolutionOutput,
    _assemble_resolution,
    _derive_resolved,
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
        ("agent_done", "resolved", True),
        ("customer_done", "customer_satisfied", True),
        ("agent_done", "issue_fixed", True),
        ("agent_done", "closed", True),
        ("agent_done", "escalation", False),
        ("customer_done", "customer_frustrated", False),
        ("agent_done", None, False),
        ("cap_hit", "resolved", False),
        ("cap_hit", None, False),
    ],
)
def test_derive_resolved(end_reason: str, last_reason: str | None, expected: bool) -> None:
    assert _derive_resolved(end_reason, last_reason) is expected


def test_assemble_resolution_builds_output() -> None:
    turns = [
        DialogueTurnOutput(speaker="customer", content="opening", done=False),
        DialogueTurnOutput(speaker="agent", content="fixed it", done=True, done_reason="resolved"),
    ]
    res = _assemble_resolution(subject="Sub", body="opening", turns=turns, end_reason="agent_done")
    assert isinstance(res, ResolutionOutput)
    assert res.subject == "Sub"
    assert res.body == "opening"
    assert res.turns == turns
    assert res.end_reason == "agent_done"
    assert res.resolved is True


def test_assemble_resolution_cap_hit_is_unresolved() -> None:
    turns = [DialogueTurnOutput(speaker="customer", content="x", done=False)]
    res = _assemble_resolution(subject="S", body="x", turns=turns, end_reason="cap_hit")
    assert res.resolved is False
