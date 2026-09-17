"""Integration: a dialogue that never flags done terminates at the hard turn cap."""

from __future__ import annotations

import asyncio
from pathlib import Path

from csfd.agents.factory import AgentFactory
from csfd.graph.phase2_graph import build_phase2_subgraph
from csfd.models.fake import FakeChatModel
from csfd.pipeline import (
    ConsistencyVerdict,
    DialogueTurnOutput,
    IncomingRequestOutput,
)
from tests.integration import _dialogue_harness as h


def _factory() -> AgentFactory:
    fake = FakeChatModel(
        structured={
            IncomingRequestOutput: IncomingRequestOutput(subject="S", body="It keeps failing."),
            # A single never-done turn is returned for every turn call.
            DialogueTurnOutput: DialogueTurnOutput(speaker="agent", content="...", done=False),
            ConsistencyVerdict: ConsistencyVerdict(status="pass"),
        }
    )
    return h.factory(fake)


def test_dialogue_cap_hit_is_unresolved(tmp_path: Path) -> None:
    db = h.setup_db(tmp_path)
    settings = h.build_settings(tmp_path, validation_enabled=True, max_retries=2, turn_cap=4)
    graph = build_phase2_subgraph(factory=_factory(), db=db, settings=settings)

    asyncio.run(graph.ainvoke(h.initial_state(settings)))

    row = h.resolution_row(db)
    assert row["turn_count"] == 4  # opening + 3 turns, then cap fires
    assert row["resolved"] is False
    assert row["quality_flag"] == "warning:turn_cap_hit"
    # Every turn call gets its own trace row, even two agent turns in one attempt.
    assert h.trace_count(db, node_name="agent_turn_generator") == 2
    assert h.trace_count(db, node_name="customer_turn_generator") == 1
