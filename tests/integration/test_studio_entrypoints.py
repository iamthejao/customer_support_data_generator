"""LangGraph Studio entrypoints + ``langgraph.json`` smoke test.

Must be run from the repo root so ``load_settings()`` can locate
``config/default.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_langgraph_json_exists_and_is_valid() -> None:
    p = Path("langgraph.json")
    assert p.exists()
    data = json.loads(p.read_text())
    assert "graphs" in data
    assert "phase1" in data["graphs"]
    assert "phase2" in data["graphs"]
    for name, ref in data["graphs"].items():
        module_path, _, _attr = ref.partition(":")
        local = module_path[2:] if module_path.startswith("./") else module_path
        assert Path(local).exists(), f"graph entry {name} -> {local} missing"


def test_studio_factory_functions_exist_and_return_graph() -> None:
    from csfd.graph.compose import (
        make_phase1_studio_graph,
        make_phase2_studio_graph,
    )

    g1 = make_phase1_studio_graph()
    g2 = make_phase2_studio_graph()
    assert hasattr(g1, "get_graph")
    assert hasattr(g2, "get_graph")
    text1 = g1.get_graph().draw_mermaid()
    text2 = g2.get_graph().draw_mermaid()
    assert "brainstorm" in text1 or "problem" in text1
    assert "turn" in text2 or "writer" in text2
