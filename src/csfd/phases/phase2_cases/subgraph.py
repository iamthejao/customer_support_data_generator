"""Phase 2 LangGraph subgraph composition."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from csfd.phases.phase2_cases.routing import (
    aggregate_turn_verdicts,
    dispatch_turn_checkers,
    route_turn_loop,
    route_turn_verdict,
)
from csfd.phases.phase2_cases.state import TicketState


def build_phase2_graph(
    *,
    max_retries: int,
    creative_noise_probability: float,  # used by extended graph in Plan 5
    min_turns: int,
    max_turns: int,
) -> CompiledStateGraph:
    """Build (and compile) the structural Phase 2 subgraph.

    Like the Phase 1 builder, this is a *structural* subgraph — Plan 5 wires
    the `AgentFactory`, `Database`, and `Phase2Config` into the nodes via
    `functools.partial` and adds the outer multi-ticket loop with the parent
    graph composition.
    """
    g = StateGraph(TicketState)

    def _placeholder(_state: TicketState) -> dict[str, object]:
        raise NotImplementedError("Bind agent factories + db at subgraph instantiation (Plan 5).")

    for n in (
        "load_kb",
        "ticket_sampler",
        "ticket_init",
        "turn_writer",
        "creative_noise_gate",
        "creative_noise",
        "turn_check_dispatch",
        "turn_consistency_check",
        "turn_background_check",
        "turn_scenario_check",
        "turn_route",
        "turn_loop",
    ):
        g.add_node(n, _placeholder)
    g.add_node("turn_aggregate", aggregate_turn_verdicts)

    g.add_edge(START, "load_kb")
    g.add_edge("load_kb", "ticket_sampler")
    g.add_edge("ticket_sampler", "ticket_init")
    g.add_edge("ticket_init", "turn_writer")
    g.add_edge("turn_writer", "creative_noise_gate")

    # Probabilistic gate decides noise vs straight-to-check.
    # Placeholder: real probability comes from cfg at instantiation in Plan 5.
    g.add_conditional_edges(
        "creative_noise_gate",
        lambda _state: "skip_noise",
        {
            "apply_noise": "creative_noise",
            "skip_noise": "turn_check_dispatch",
        },
    )
    g.add_edge("creative_noise", "turn_check_dispatch")

    g.add_conditional_edges(
        "turn_check_dispatch",
        dispatch_turn_checkers,  # type: ignore[arg-type]
        {
            "turn_consistency_check": "turn_consistency_check",
            "turn_background_check": "turn_background_check",
            "turn_scenario_check": "turn_scenario_check",
        },
    )
    g.add_edge("turn_consistency_check", "turn_aggregate")
    g.add_edge("turn_background_check", "turn_aggregate")
    g.add_edge("turn_scenario_check", "turn_aggregate")

    def _route_verdict(state: TicketState) -> str:
        return route_turn_verdict(state, max_retries=max_retries)

    g.add_edge("turn_aggregate", "turn_route")
    g.add_conditional_edges(
        "turn_route",
        _route_verdict,
        {
            "commit": "turn_loop",
            "commit_with_warning": "turn_loop",
            "regenerate": "turn_writer",
        },
    )

    def _route_loop(state: TicketState) -> str:
        return route_turn_loop(state, min_turns=min_turns, max_turns=max_turns)

    g.add_conditional_edges(
        "turn_loop",
        _route_loop,
        {"more_turns": "turn_writer", "ticket_done": END},
    )

    return g.compile()
