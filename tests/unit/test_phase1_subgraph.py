import pytest

from csfd.phases.phase1_kb.subgraph import build_phase1_graph


def test_build_phase1_graph_compiles() -> None:
    graph = build_phase1_graph(
        max_retries=3,
        dedup_threshold=0.85,
        kb_target_rate=0.7,
    )
    nodes = graph.get_graph().nodes
    expected = {
        "seed_load",
        "problem_brainstorm",
        "problem_consistency_check",
        "problem_background_check",
        "problem_scenario_check",
        "problem_aggregate",
    }
    for n in expected:
        assert n in nodes


@pytest.mark.asyncio
async def test_build_phase1_graph_renders_mermaid_without_error() -> None:
    graph = build_phase1_graph(
        max_retries=3,
        dedup_threshold=0.85,
        kb_target_rate=0.7,
    )
    text = graph.get_graph().draw_mermaid()
    assert "problem_brainstorm" in text
