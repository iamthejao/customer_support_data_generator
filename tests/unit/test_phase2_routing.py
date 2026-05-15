from uuid import uuid4

from langgraph.types import Send

from csfd.agents.base import Issue, Verdict
from csfd.phases.phase2_cases.routing import (
    aggregate_turn_verdicts,
    dispatch_turn_checkers,
    route_turn_loop,
    route_turn_verdict,
)
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CommittedTurn,
    CustomerPersona,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.seeds.company import CompanyProfile


def _state(
    *,
    verdicts: list[Verdict] | None = None,
    retry: int = 0,
    turn_index: int = 0,
    turns_committed: int = 0,
) -> TicketState:
    ticket = CommittedTicket(
        id="t-1",
        draft=TicketDraft(
            problem_id="p-1",
            kb_article_id=None,
            ticket_type="l1",
            priority="medium",
            subject="s",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="n"),
            agent_persona=AgentPersona(name="A", tier="l1", expertise=""),
        ),
    )
    return TicketState(
        run_id=str(uuid4()),
        parent_run_id=str(uuid4()),
        run_seed=1,
        company=CompanyProfile(name="A", raw_markdown="# A"),
        current_ticket=ticket,
        current_turn_draft=TurnDraft(
            speaker="agent",
            content="x",
            intent="question",
        ),
        verdicts=verdicts or [],
        retry_attempt=retry,
        turn_index=turn_index,
        turns_committed=[
            CommittedTurn(
                id=str(i),
                ticket_id="t-1",
                turn_index=i,
                draft=TurnDraft(speaker="agent", content="x", intent="question"),
            )
            for i in range(turns_committed)
        ],
    )


def test_dispatch_turn_checkers_sends_to_three_nodes() -> None:
    state = _state()
    sends = dispatch_turn_checkers(state)
    targets = sorted(s.node for s in sends)
    assert targets == [
        "turn_background_check",
        "turn_consistency_check",
        "turn_scenario_check",
    ]
    for s in sends:
        assert isinstance(s, Send)


def test_aggregate_turn_verdicts_is_noop_passthrough() -> None:
    state = _state(verdicts=[Verdict(checker="x", passed=True)])
    update = aggregate_turn_verdicts(state)
    assert update == {}


def test_route_turn_verdict_all_pass_returns_commit() -> None:
    state = _state(
        verdicts=[
            Verdict(checker="consistency", passed=True),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ]
    )
    assert route_turn_verdict(state, max_retries=3) == "commit"


def test_route_turn_verdict_fail_under_budget_returns_regenerate() -> None:
    state = _state(
        verdicts=[
            Verdict(
                checker="consistency",
                passed=False,
                issues=[
                    Issue(severity="error", location="x", rule_violated="r", explanation="e"),
                ],
            ),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ],
        retry=1,
    )
    assert route_turn_verdict(state, max_retries=3) == "regenerate"


def test_route_turn_verdict_fail_over_budget_returns_commit_with_warning() -> None:
    state = _state(
        verdicts=[
            Verdict(
                checker="consistency",
                passed=False,
                issues=[
                    Issue(severity="error", location="x", rule_violated="r", explanation="e"),
                ],
            ),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ],
        retry=3,
    )
    assert route_turn_verdict(state, max_retries=3) == "commit_with_warning"


def test_route_turn_loop_more_turns_when_under_max() -> None:
    state = _state(turns_committed=2)
    assert route_turn_loop(state, min_turns=2, max_turns=6) == "more_turns"


def test_route_turn_loop_done_when_at_max() -> None:
    state = _state(turns_committed=6)
    assert route_turn_loop(state, min_turns=2, max_turns=6) == "ticket_done"


def test_route_turn_loop_more_turns_below_min() -> None:
    state = _state(turns_committed=1)
    assert route_turn_loop(state, min_turns=2, max_turns=6) == "more_turns"
