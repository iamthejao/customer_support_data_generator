"""Phase 1 LangGraph subgraph composition (structural; runtime wiring in Plan 5)."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from csfd.phases.phase1_kb.routing import (
    aggregate_verdicts,
    dispatch_problem_checkers,
    route_problem_verdict,
)
from csfd.phases.phase1_kb.state import KBState


def build_phase1_graph(
    *,
    max_retries: int,
    dedup_threshold: float,  # used by extended graph in Plan 5
    kb_target_rate: float,  # used by extended graph in Plan 5
) -> CompiledStateGraph:
    """Build (and compile) the minimal Phase 1 subgraph.

    NOTE: brainstorm/check nodes are async functions taking (state, **kwargs).
    LangGraph injects only `state`; per-node kwargs (factory) are bound at
    subgraph instantiation in Plan 5 via functools.partial. Here the nodes are
    placeholder functions that raise NotImplementedError — production usage is
    via the parent graph in Plan 5.
    """
    g = StateGraph(KBState)

    def _placeholder(_state: KBState) -> dict[str, Any]:
        raise NotImplementedError("Bind agent factories at subgraph instantiation (see Plan 5).")

    async def _seed_load_placeholder(_state: KBState) -> dict[str, Any]:
        return {}

    g.add_node("seed_load", _seed_load_placeholder)
    g.add_node("problem_brainstorm", _placeholder)
    g.add_node("problem_consistency_check", _placeholder)
    g.add_node("problem_background_check", _placeholder)
    g.add_node("problem_scenario_check", _placeholder)
    g.add_node("problem_aggregate", aggregate_verdicts)

    g.add_edge(START, "seed_load")
    g.add_edge("seed_load", "problem_brainstorm")

    g.add_conditional_edges(
        "problem_brainstorm",
        dispatch_problem_checkers,  # type: ignore[arg-type]
        {
            "problem_consistency_check": "problem_consistency_check",
            "problem_background_check": "problem_background_check",
            "problem_scenario_check": "problem_scenario_check",
        },
    )
    g.add_edge("problem_consistency_check", "problem_aggregate")
    g.add_edge("problem_background_check", "problem_aggregate")
    g.add_edge("problem_scenario_check", "problem_aggregate")

    def _route(state: KBState) -> str:
        return route_problem_verdict(state, max_retries=max_retries)

    g.add_conditional_edges(
        "problem_aggregate",
        _route,
        {
            "commit": END,
            "commit_with_warning": END,
            "regenerate": "problem_brainstorm",
        },
    )

    return g.compile()
