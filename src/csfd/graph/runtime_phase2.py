"""Runtime-wired Phase 2 LangGraph — single-turn subgraph that runs end-to-end.

This module composes the Phase 2 turn-generation nodes (defined in
``csfd.phases.phase2_cases.nodes``) into a fully runnable LangGraph by binding
the ``AgentFactory`` and ``Database`` dependencies via ``functools.partial``.

The graph processes ONE turn end-to-end:
    turn_writer -> creative-noise gate (apply or skip) -> 3 parallel checkers
    -> commit (persist) -> route (commit / commit_with_warning end the graph,
    regenerate loops back to turn_writer with retry counter bumped).

The outer per-ticket turn loop and per-problem ticket loop are implemented
separately in ``compose.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from csfd.agents.factory import AgentFactory
from csfd.phases.phase2_cases.nodes import (
    creative_noise_gate,
    creative_noise_node,
    persist_ticket_node,
    persist_turn_node,
    turn_background_check_node,
    turn_consistency_check_node,
    turn_scenario_check_node,
    turn_writer_node,
)
from csfd.phases.phase2_cases.routing import (
    route_turn_verdict,
)
from csfd.phases.phase2_cases.state import CommittedTurn, TicketState
from csfd.storage.db import Database


def _ensure_ticket(state: TicketState, *, db: Database) -> dict[str, Any]:
    """Idempotently persist ``state.current_ticket`` before any turns.

    ``TurnRecord.ticket_id`` is a SQLite FK to ``tickets.id``, so the ticket
    row must exist before the first turn is committed. ``TicketRepo.create``
    is ``INSERT OR IGNORE`` so re-running this on the retry loop is a no-op.
    Always returns at least one state-schema field so LangGraph's
    "must write" rule is satisfied even on the no-op branch (no ticket).
    """
    if state.current_ticket is None:
        return {"current_ticket": state.current_ticket}
    persist_ticket_node(state, db=db, ticket=state.current_ticket)
    return {"current_ticket": state.current_ticket}


def _route_turn(state: TicketState, *, max_retries: int) -> str:
    """Route looking only at the most recent 3 verdicts.

    Verdicts accumulate across retries (operator.add reducer), so we slice
    the last three (one per checker for the most-recent attempt) before
    delegating to the underlying ``route_turn_verdict``.
    """
    last3 = state.verdicts[-3:] if len(state.verdicts) >= 3 else state.verdicts
    snapshot = state.model_copy(update={"verdicts": last3})
    return route_turn_verdict(snapshot, max_retries=max_retries)


def _bump_retry(state: TicketState) -> dict[str, Any]:
    """Increment the retry counter before looping back to ``turn_writer``."""
    return {"retry_attempt": state.retry_attempt + 1}


def _gate(state: TicketState, *, probability: float) -> str:
    """Delegate to the noise gate routing helper."""
    return creative_noise_gate(state, probability=probability)


def _aggregate_turn_verdicts_node(state: TicketState) -> dict[str, Any]:
    """Fan-in barrier for the noise gate (apply/skip) branches.

    LangGraph 0.2.28 requires every node to write at least one state-schema
    field, so we re-write ``current_turn_draft`` (a regular non-reducer
    field) back to itself as an idempotent no-op state update. This node is
    also the source of the Send fan-out to the 3 turn checkers.
    """
    return {"current_turn_draft": state.current_turn_draft}


def _dispatch_turn_checkers(state: TicketState) -> list[Send]:
    """Fan out the candidate turn to the 3 checker nodes.

    Mirrors ``csfd.phases.phase2_cases.routing.dispatch_turn_checkers`` but
    serialises the Pydantic payloads with ``.model_dump()`` so the downstream
    Jinja2 prompt templates (which use the ``tojson`` filter on ``inputs``)
    can render without a JSON-serialisation error on raw Pydantic models.
    """
    payload: dict[str, Any] = {
        "current_turn_draft": (
            state.current_turn_draft.model_dump() if state.current_turn_draft is not None else None
        ),
        "current_ticket": (
            state.current_ticket.model_dump() if state.current_ticket is not None else None
        ),
    }
    return [
        Send("turn_consistency_check", payload),
        Send("turn_background_check", payload),
        Send("turn_scenario_check", payload),
    ]


async def _commit_turn(state: TicketState, *, db: Database) -> dict[str, Any]:
    """Move the current draft into ``turns_committed`` and persist it.

    Guards against an empty draft / missing ticket by re-writing the current
    draft back to itself (no-op) so LangGraph's "must write" rule is
    satisfied even on the no-op path.
    """
    if state.current_turn_draft is None or state.current_ticket is None:
        # No-op write — re-affirm current value to satisfy LangGraph's "must write" rule.
        return {"current_turn_draft": state.current_turn_draft}
    committed = CommittedTurn(
        id=str(uuid4()),
        ticket_id=state.current_ticket.id,
        turn_index=state.turn_index,
        draft=state.current_turn_draft,
    )
    persist_turn_node(state, db=db, turn=committed)
    return {
        "turns_committed": [committed],
        "turn_index": state.turn_index + 1,
    }


def build_runtime_phase2_graph(
    *,
    factory: AgentFactory,
    db: Database,
    max_retries: int,
    noise_probability: float,
    noise_type_weights: Mapping[str, float],
) -> CompiledStateGraph:
    """Build (and compile) a runtime Phase 2 graph for a single turn.

    Args:
        factory: ``AgentFactory`` producing the turn writer, the 3 turn
            checkers, and the creative-noise agent.
        db: SQLite ``Database`` for persisting committed turns.
        max_retries: Maximum writer retries before forcibly committing a turn
            with a quality warning.
        noise_probability: Probability (0.0-1.0) of applying CreativeNoise to
            a turn before checking.
        noise_type_weights: Weighted distribution of noise types passed to the
            creative-noise agent.

    Returns:
        A compiled LangGraph ready to be ``ainvoke``d with a ``TicketState``.
    """
    g: StateGraph = StateGraph(TicketState)

    g.add_node("ensure_ticket", partial(_ensure_ticket, db=db))
    g.add_node("turn_writer", partial(turn_writer_node, factory=factory))
    g.add_node(
        "creative_noise",
        partial(
            creative_noise_node,
            factory=factory,
            noise_type_weights=noise_type_weights,
        ),
    )
    g.add_node(
        "turn_consistency_check",
        partial(turn_consistency_check_node, factory=factory),
    )
    g.add_node(
        "turn_background_check",
        partial(turn_background_check_node, factory=factory),
    )
    g.add_node(
        "turn_scenario_check",
        partial(turn_scenario_check_node, factory=factory),
    )
    g.add_node("turn_aggregate", _aggregate_turn_verdicts_node)
    g.add_node("bump_retry", _bump_retry)
    g.add_node("commit_turn", partial(_commit_turn, db=db))

    g.add_edge(START, "ensure_ticket")
    g.add_edge("ensure_ticket", "turn_writer")
    g.add_conditional_edges(
        "turn_writer",
        partial(_gate, probability=noise_probability),
        {"apply_noise": "creative_noise", "skip_noise": "turn_aggregate"},
    )
    g.add_edge("creative_noise", "turn_aggregate")

    # ``turn_aggregate`` also serves as the Send dispatch entrypoint — the 3
    # checkers run in parallel and their verdicts converge via the
    # Annotated[list, operator.add] reducer on TicketState.verdicts.
    g.add_conditional_edges(
        "turn_aggregate",
        _dispatch_turn_checkers,  # type: ignore[arg-type]
        {
            "turn_consistency_check": "turn_consistency_check",
            "turn_background_check": "turn_background_check",
            "turn_scenario_check": "turn_scenario_check",
        },
    )
    g.add_edge("turn_consistency_check", "commit_turn")
    g.add_edge("turn_background_check", "commit_turn")
    g.add_edge("turn_scenario_check", "commit_turn")

    def _route(state: TicketState) -> str:
        return _route_turn(state, max_retries=max_retries)

    g.add_conditional_edges(
        "commit_turn",
        _route,
        {
            "commit": END,
            "commit_with_warning": END,
            "regenerate": "bump_retry",
        },
    )
    g.add_edge("bump_retry", "turn_writer")

    return g.compile()
