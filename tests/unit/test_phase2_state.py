from typing import get_args, get_type_hints
from uuid import uuid4

from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CommittedTurn,
    CustomerPersona,
    Phase2Stats,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.seeds.company import CompanyProfile


def _empty_company() -> CompanyProfile:
    return CompanyProfile(name="Acme", raw_markdown="# Acme")


def test_ticket_state_defaults() -> None:
    state = TicketState(
        run_id=str(uuid4()),
        parent_run_id=str(uuid4()),
        run_seed=42,
        company=_empty_company(),
    )
    assert state.problem_pool == []
    assert state.kb_by_problem == {}
    assert state.current_ticket is None
    assert state.turns_committed == []
    assert state.current_turn_draft is None
    assert state.noise_applied is False
    assert state.noise_type is None
    assert state.verdicts == []
    assert state.retry_attempt == 0
    assert state.turn_index == 0
    assert isinstance(state.stats, Phase2Stats)


def test_ticket_state_verdict_reducer_is_operator_add() -> None:
    import operator

    hints = get_type_hints(TicketState, include_extras=True)
    annotated = hints["verdicts"]
    args = get_args(annotated)
    assert args[1] is operator.add


def test_turn_draft_required_fields() -> None:
    t = TurnDraft(
        speaker="agent",
        content="Hello, how can I help?",
        intent="question",
    )
    assert t.speaker == "agent"
    assert t.kb_references == []
    assert t.noise_applied is False


def test_ticket_draft_required_fields() -> None:
    td = TicketDraft(
        problem_id="p1",
        kb_article_id=None,
        ticket_type="l3",
        priority="high",
        subject="Cannot log in",
        customer_persona=CustomerPersona(name="C", tier="standard", tone="frustrated"),
        agent_persona=AgentPersona(name="A", tier="l3", expertise="auth specialist"),
    )
    assert td.ticket_type == "l3"
    assert td.kb_article_id is None


def test_committed_ticket_and_turn() -> None:
    ct = CommittedTicket(
        id="t1",
        draft=TicketDraft(
            problem_id="p1",
            kb_article_id="a1",
            ticket_type="l1",
            priority="medium",
            subject="s",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="neutral"),
            agent_persona=AgentPersona(name="A", tier="l1", expertise=""),
        ),
        status="resolved",
    )
    assert ct.draft.ticket_type == "l1"

    cu = CommittedTurn(
        id="tu1",
        ticket_id="t1",
        turn_index=0,
        draft=TurnDraft(speaker="customer", content="help", intent="question"),
    )
    assert cu.turn_index == 0
