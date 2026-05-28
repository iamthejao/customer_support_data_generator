"""Integration: a pass-with-edits consistency verdict rewrites the persisted transcript."""

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

_EDITED = [
    DialogueTurnOutput(speaker="customer", content="It power-cycles.", done=False),
    DialogueTurnOutput(speaker="agent", content="Reseat the connector please.", done=False),
    DialogueTurnOutput(
        speaker="customer",
        content="Done — resolved, thank you.",
        done=True,
        done_reason="customer_satisfied",
    ),
]


def _factory() -> AgentFactory:
    fake = FakeChatModel(
        structured={
            IncomingRequestOutput: IncomingRequestOutput(subject="S", body="It power-cycles."),
            ConsistencyVerdict: ConsistencyVerdict(
                status="pass_with_edits",
                issues=["smoothed wording"],
                edited_subject="Power cycling resolved",
                edited_body="It power-cycles.",
                edited_turns=_EDITED,
            ),
        },
        structured_seq={
            DialogueTurnOutput: [
                DialogueTurnOutput(speaker="agent", content="reseat it", done=False),
                DialogueTurnOutput(
                    speaker="customer", content="fixed", done=True, done_reason="customer_satisfied"
                ),
            ],
        },
    )
    return h.factory(fake)


def test_dialogue_consistency_pass_with_edits(tmp_path: Path) -> None:
    db = h.setup_db(tmp_path)
    settings = h.build_settings(tmp_path, validation_enabled=True, max_retries=2)
    graph = build_phase2_subgraph(factory=_factory(), db=db, settings=settings)

    asyncio.run(graph.ainvoke(h.initial_state(settings)))

    row = h.resolution_row(db)
    assert row["turn_count"] == 3
    assert row["turns"][1]["content"] == "Reseat the connector please."
    assert row["quality_flag"] == "info:consistency_edited"
    assert row["resolved"] is True
