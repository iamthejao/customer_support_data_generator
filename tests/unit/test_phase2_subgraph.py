import pytest

from csfd.phases.phase2_cases.subgraph import build_phase2_graph


def test_build_phase2_graph_compiles_with_expected_nodes() -> None:
    graph = build_phase2_graph(
        max_retries=3,
        creative_noise_probability=0.2,
        min_turns=2,
        max_turns=6,
    )
    nodes = graph.get_graph().nodes
    expected = {
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
        "turn_aggregate",
        "turn_route",
        "turn_loop",
    }
    for n in expected:
        assert n in nodes, f"missing node: {n}"


@pytest.mark.asyncio
async def test_build_phase2_graph_renders_mermaid() -> None:
    graph = build_phase2_graph(
        max_retries=3,
        creative_noise_probability=0.2,
        min_turns=2,
        max_turns=6,
    )
    text = graph.get_graph().draw_mermaid()
    assert "turn_writer" in text
    assert "creative_noise" in text
