"""Unit tests for ConsistencyVerdict and edit application."""

from __future__ import annotations

from csfd.pipeline import (
    ConsistencyVerdict,
    DialogueTurnOutput,
    ResolutionOutput,
    _apply_consistency_edits,
)


def _draft() -> ResolutionOutput:
    return ResolutionOutput(
        subject="orig subj",
        body="orig body",
        turns=[
            DialogueTurnOutput(speaker="customer", content="orig body", done=False),
            DialogueTurnOutput(speaker="agent", content="ok", done=True, done_reason="resolved"),
        ],
        resolved=True,
        end_reason="agent_done",
    )


def test_pass_returns_draft_unchanged() -> None:
    draft = _draft()
    verdict = ConsistencyVerdict(status="pass")
    out = _apply_consistency_edits(draft, verdict)
    assert out is draft


def test_pass_with_edits_replaces_provided_fields() -> None:
    draft = _draft()
    new_turns = [
        DialogueTurnOutput(speaker="customer", content="edited body", done=False),
        DialogueTurnOutput(speaker="agent", content="edited", done=True, done_reason="resolved"),
    ]
    verdict = ConsistencyVerdict(
        status="pass_with_edits",
        edited_subject="new subj",
        edited_body="edited body",
        edited_turns=new_turns,
    )
    out = _apply_consistency_edits(draft, verdict)
    assert out.subject == "new subj"
    assert out.body == "edited body"
    assert out.turns == new_turns
    assert out.resolved is True
    assert out.end_reason == "agent_done"


def test_pass_with_edits_partial_keeps_originals() -> None:
    draft = _draft()
    verdict = ConsistencyVerdict(status="pass_with_edits", edited_subject="only subj")
    out = _apply_consistency_edits(draft, verdict)
    assert out.subject == "only subj"
    assert out.body == "orig body"
    assert out.turns == draft.turns


def test_appended_closing_turn_moves_end_reason_to_last_speaker() -> None:
    # Seen live: the checker appended an agent wrap-up after the customer's done turn.
    draft = _draft().model_copy(update={"end_reason": "customer_done"})
    edited = [
        *draft.turns,
        DialogueTurnOutput(
            speaker="agent", content="Glad it works.", done=True, done_reason="resolved"
        ),
    ]
    out = _apply_consistency_edits(
        draft, ConsistencyVerdict(status="pass_with_edits", edited_turns=edited)
    )
    assert out.end_reason == "agent_done"
    assert out.resolved is True


def test_edits_keep_cap_and_drop_end_reasons() -> None:
    for reason in ("cap_hit", "dropped"):
        draft = _draft().model_copy(update={"end_reason": reason})
        out = _apply_consistency_edits(
            draft, ConsistencyVerdict(status="pass_with_edits", edited_turns=draft.turns)
        )
        assert out.end_reason == reason
        assert out.resolved is False
