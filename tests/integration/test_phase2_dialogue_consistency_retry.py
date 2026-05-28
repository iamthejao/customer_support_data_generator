"""Integration: a consistency failure re-rolls the whole conversation, then passes."""

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
            IncomingRequestOutput: IncomingRequestOutput(subject="S", body="It power-cycles."),
        },
        structured_seq={
            # Two full rolls: each agent turn flags done immediately (2-turn conversation).
            DialogueTurnOutput: [
                DialogueTurnOutput(
                    speaker="agent", content="Reseat it.", done=True, done_reason="resolved"
                ),
                DialogueTurnOutput(
                    speaker="agent", content="Reseat it (again).", done=True, done_reason="resolved"
                ),
            ],
            # First review fails (re-roll), second passes.
            ConsistencyVerdict: [
                ConsistencyVerdict(status="fail", issues=["agent jumped to the fix too fast"]),
                ConsistencyVerdict(status="pass"),
            ],
        },
    )
    return h.factory(fake)


def test_dialogue_consistency_fail_rerolls_then_commits(tmp_path: Path) -> None:
    db = h.setup_db(tmp_path)
    settings = h.build_settings(tmp_path, validation_enabled=True, max_retries=1)
    graph = build_phase2_subgraph(factory=_factory(), db=db, settings=settings)

    asyncio.run(graph.ainvoke(h.initial_state(settings)))

    from csfd.storage.repository import ResolutionRepo

    # Only the final, passing conversation commits.
    assert ResolutionRepo(db).count_for_run(h.RUN_ID) == 1
    row = h.resolution_row(db)
    assert row["resolved"] is True
    # Consistency ran twice: one fail (re-roll) + one pass.
    assert h.trace_count(db, node_name="conversation_consistency_check") == 2
