"""Integration: seeded case facts reach both speakers, the checker and the lineage row.

A transcript that names a machine other than the case record's fails the
code-level identifier check even when the LLM checker passes it, and re-rolls
through the ordinary consistency retry path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from csfd.agents.factory import AgentFactory
from csfd.graph.phase2_graph import build_phase2_subgraph
from csfd.models.fake import FakeChatModel
from csfd.pipeline import ConsistencyVerdict, DialogueTurnOutput, IncomingRequestOutput
from csfd.seeds.company import AssetModel, CaseFactsCatalogue, CompanyProfile
from csfd.storage.db import Database
from tests.integration import _dialogue_harness as h

COMPANY = CompanyProfile(
    name="Acme",
    raw_markdown="# Acme",
    case_facts=CaseFactsCatalogue(
        assets=[AssetModel("AX-100", "Edge appliance", "AX100-#####-?")],
        caller_roles=["plant engineer"],
        site_locales=["en_GB"],
    ),
)


def _factory(first_reply: str, verdicts: int) -> AgentFactory:
    fake = FakeChatModel(
        structured={
            IncomingRequestOutput: IncomingRequestOutput(subject="S", body="It power-cycles."),
        },
        structured_seq={
            DialogueTurnOutput: [
                DialogueTurnOutput(
                    speaker="agent", content=first_reply, done=True, done_reason="resolved"
                ),
                DialogueTurnOutput(
                    speaker="agent",
                    content="Reseat the connector on your AX-100.",
                    done=True,
                    done_reason="resolved",
                ),
            ],
            # The LLM checker passes every attempt; only the code check can fail one.
            ConsistencyVerdict: [ConsistencyVerdict(status="pass")] * verdicts,
        },
    )
    return h.factory(fake)


def _inputs(db: Database, node_name: str) -> list[dict[str, Any]]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT input_json FROM agent_traces WHERE run_id = ? AND node_name = ? ORDER BY rowid",
            (h.RUN_ID, node_name),
        ).fetchall()
    return [json.loads(r["input_json"]) for r in rows]


def _lineage_facts(db: Database) -> dict[str, Any]:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT case_facts_json FROM lineage WHERE run_id = ?", (h.RUN_ID,)
        ).fetchone()
    return dict(json.loads(row["case_facts_json"]))


def test_wrong_model_rerolls_and_facts_reach_every_speaker(tmp_path: Path) -> None:
    db = h.setup_db(tmp_path)
    settings = h.build_settings(tmp_path, validation_enabled=True, max_retries=1)
    graph = build_phase2_subgraph(
        factory=_factory("Reseat the connector on your AX-400.", verdicts=2),
        db=db,
        settings=settings,
    )

    asyncio.run(graph.ainvoke(h.initial_state(settings, COMPANY)))

    facts = _lineage_facts(db)
    assert facts["asset_model"] == "AX-100"
    assert facts["asset_serial"].startswith("AX100-")
    assert facts["caller_role"] == "plant engineer"
    assert facts["site_country"] == "United Kingdom"

    # The first attempt named the AX-400, so it re-rolled; the second one commits clean.
    assert h.trace_count(db, node_name="conversation_consistency_check") == 2
    row = h.resolution_row(db)
    assert row["quality_flag"] is None
    assert row["turns"][-1]["content"] == "Reseat the connector on your AX-100."

    customer = _inputs(db, "incoming_request_generator")
    assert customer[0]["facts"]["asset_serial"] == facts["asset_serial"]
    assert customer[0]["facts"]["role"] == "plant engineer"
    assert "root_cause" not in json.dumps(customer[0])
    # The re-roll tells the customer which identifier went wrong.
    assert customer[0]["fact_issues"] == []
    assert any("AX-400" in i for i in customer[1]["fact_issues"])

    agent = _inputs(db, "agent_turn_generator")
    assert agent[0]["facts"]["asset_model"] == "AX-100"
    assert any("AX-400" in i for i in agent[1]["prior_issues"])

    checker = _inputs(db, "conversation_consistency_check")[0]
    assert checker["facts"]["asset_serial"] == facts["asset_serial"]
    assert checker["problem"]["background"] == h.problem().background


def test_wrong_model_with_no_retries_left_commits_flagged(tmp_path: Path) -> None:
    db = h.setup_db(tmp_path)
    settings = h.build_settings(tmp_path, validation_enabled=True, max_retries=0)
    graph = build_phase2_subgraph(
        factory=_factory("It's the AX-400, isn't it?", verdicts=1), db=db, settings=settings
    )

    asyncio.run(graph.ainvoke(h.initial_state(settings, COMPANY)))

    assert h.resolution_row(db)["quality_flag"] == "warning:retries_exhausted"
