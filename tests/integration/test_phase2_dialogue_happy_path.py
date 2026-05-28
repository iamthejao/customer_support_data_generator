"""Integration: a scripted dialogue ends when the customer flags done; one slot commits."""

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
            IncomingRequestOutput: IncomingRequestOutput(
                subject="Power", body="It power-cycles every 20 minutes."
            ),
            ConsistencyVerdict: ConsistencyVerdict(status="pass"),
        },
        structured_seq={
            DialogueTurnOutput: [
                DialogueTurnOutput(
                    speaker="agent", content="Please reseat the power connector.", done=False
                ),
                DialogueTurnOutput(
                    speaker="customer",
                    content="That fixed it, thank you!",
                    done=True,
                    done_reason="customer_satisfied",
                ),
            ],
        },
    )
    return h.factory(fake)


def test_dialogue_happy_path_commits_resolved(tmp_path: Path) -> None:
    db = h.setup_db(tmp_path)
    settings = h.build_settings(tmp_path, validation_enabled=True, max_retries=2)
    graph = build_phase2_subgraph(factory=_factory(), db=db, settings=settings)

    asyncio.run(graph.ainvoke(h.initial_state(settings)))

    row = h.resolution_row(db)
    assert row["turn_count"] == 3  # opening customer + agent + customer
    assert row["resolved"] is True
    assert row["turns"][0]["speaker"] == "customer"
    assert row["turns"][-1]["done"] is True
    assert row["turns"][-1]["done_reason"] == "customer_satisfied"

    # incoming(1) + agent_turn(1) + customer_turn(1) + consistency(1) == 4 traces for the slot
    assert h.trace_count(db) == 4
    assert h.trace_count(db, node_name="conversation_consistency_check") == 1
