"""Routing helpers for the Phase 2 LangGraph subgraph."""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from csfd.phases.phase2_cases.state import TicketState


def dispatch_turn_checkers(state: TicketState) -> list[Send]:
    """Fan out the candidate turn to the 3 checkers."""
    payload = {
        "current_turn_draft": state.current_turn_draft,
        "current_ticket": state.current_ticket,
    }
    return [
        Send("turn_consistency_check", payload),
        Send("turn_background_check", payload),
        Send("turn_scenario_check", payload),
    ]


def aggregate_turn_verdicts(state: TicketState) -> dict[str, Any]:
    """No-op fan-in barrier — the Annotated reducer has already accumulated the
    list. Present as a single seam where quorum / weighted-scoring logic could
    be inserted later.
    """
    return {}


def route_turn_verdict(state: TicketState, *, max_retries: int) -> str:
    """Decide what happens after the 3 turn checkers' verdicts are aggregated."""
    if all(v.passed for v in state.verdicts):
        return "commit"
    if state.retry_attempt >= max_retries:
        return "commit_with_warning"
    return "regenerate"


def route_turn_loop(
    state: TicketState,
    *,
    min_turns: int,
    max_turns: int,
) -> str:
    """Decide whether to generate another turn or close the ticket.

    Below ``max_turns``: continue. At/above ``max_turns``: stop. Plan 5 may
    swap in an LLM-judge to decide between ``min_turns`` and ``max_turns``.
    """
    n = len(state.turns_committed)
    if n >= max_turns:
        return "ticket_done"
    return "more_turns"
