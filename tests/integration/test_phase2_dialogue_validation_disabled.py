"""Integration: with validation disabled, the dialogue turns are still persisted.

Regression guard: transcript assembly must happen on the validation-skipped
path too, not only inside the consistency node.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from csfd.agents.factory import AgentFactory
from csfd.graph.phase2_graph import build_phase2_subgraph
from csfd.models.fake import FakeChatModel
from csfd.pipeline import DialogueTurnOutput, IncomingRequestOutput
from tests.integration import _dialogue_harness as h


def _factory() -> AgentFactory:
    fake = FakeChatModel(
        structured={
            IncomingRequestOutput: IncomingRequestOutput(subject="S", body="It power-cycles."),
        },
        structured_seq={
            DialogueTurnOutput: [
                DialogueTurnOutput(
                    speaker="agent",
                    content="Reseat the connector.",
                    done=True,
                    done_reason="resolved",
                ),
            ],
        },
    )
    return h.factory(fake)


def test_validation_disabled_still_persists_turns(tmp_path: Path) -> None:
    db = h.setup_db(tmp_path)
    settings = h.build_settings(tmp_path, validation_enabled=False)
    graph = build_phase2_subgraph(factory=_factory(), db=db, settings=settings)

    asyncio.run(graph.ainvoke(h.initial_state(settings)))

    row = h.resolution_row(db)
    assert row["turn_count"] == 2  # opening customer + agent done
    assert [t["speaker"] for t in row["turns"]] == ["customer", "agent"]
    assert row["resolved"] is True
    assert row["quality_flag"] == "warning:validation_skipped"
    # The consistency agent must NOT have run.
    assert h.trace_count(db, node_name="conversation_consistency_check") == 0
