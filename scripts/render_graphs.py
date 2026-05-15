"""Generate mermaid diagrams from the structural Phase 1/2 subgraphs."""

from __future__ import annotations

from pathlib import Path

from csfd.phases.phase1_kb.subgraph import build_phase1_graph
from csfd.phases.phase2_cases.subgraph import build_phase2_graph


def render_all(out_dir: Path) -> dict[str, Path]:
    """Render mermaid diagrams for Phase 1 and Phase 2 structural subgraphs.

    Returns a mapping of graph name → path written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    g1 = build_phase1_graph(max_retries=3, dedup_threshold=0.85, kb_target_rate=0.7)
    p1 = out_dir / "phase1.mmd"
    p1.write_text(g1.get_graph().draw_mermaid(), encoding="utf-8")
    written["phase1"] = p1

    g2 = build_phase2_graph(
        max_retries=3,
        creative_noise_probability=0.2,
        min_turns=2,
        max_turns=6,
    )
    p2 = out_dir / "phase2.mmd"
    p2.write_text(g2.get_graph().draw_mermaid(), encoding="utf-8")
    written["phase2"] = p2

    return written
