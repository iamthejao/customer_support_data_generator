"""Integration: the agent diagnoses from a guide, and outcomes are planned and checked.

The harness problem has a diagnosis plan (two checks, the second confirms the
cause). The agent's prompt must carry the checks but never the root cause or
the customer's findings; the customer's prompt carries the findings. The case
is planned to end in a problem state, the checker's reported state must match
it, and the committed row keeps state, ending and commitments next to the plan.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from csfd.graph.phase2_graph import build_phase2_subgraph
from csfd.models.fake import FakeChatModel
from csfd.outcomes import Commitment, ProblemState
from csfd.pipeline import ConsistencyVerdict, DialogueTurnOutput, IncomingRequestOutput
from csfd.storage.db import Database
from tests.integration import _dialogue_harness as h

_PART = Commitment(who="agent", what="ship a new PSU cable", due="tomorrow")


def _turns() -> list[DialogueTurnOutput]:
    return [
        DialogueTurnOutput(speaker="agent", content="Could you wiggle the PSU plug?"),
        DialogueTurnOutput(speaker="customer", content="The display flickers when it moves."),
        DialogueTurnOutput(
            speaker="agent", content="That points to the connector.", diagnosis_done=True
        ),
        DialogueTurnOutput(speaker="customer", content="So what now?"),
        DialogueTurnOutput(
            speaker="agent",
            content="I'll ship a new PSU cable for tomorrow.",
            done=True,
            done_reason="follow_up",
            commitments=[_PART],
        ),
    ]


def _run(tmp_path: Path, verdicts: list[ConsistencyVerdict], max_retries: int = 1) -> Database:
    fake = FakeChatModel(
        structured={IncomingRequestOutput: IncomingRequestOutput(subject="S", body="It restarts.")},
        structured_seq={
            DialogueTurnOutput: _turns() * (max_retries + 1),
            ConsistencyVerdict: list(verdicts),
        },
    )
    db = h.setup_db(tmp_path)
    settings = h.build_settings(
        tmp_path, validation_enabled=True, max_retries=max_retries, outcome="pending_part"
    )
    graph = build_phase2_subgraph(factory=h.factory(fake), db=db, settings=settings)
    asyncio.run(graph.ainvoke(h.initial_state(settings)))
    return db


def _trace_inputs(db: Database, node_name: str) -> list[dict[str, Any]]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT input_json FROM agent_traces WHERE run_id = ? AND node_name = ? "
            "ORDER BY created_at",
            (h.RUN_ID, node_name),
        ).fetchall()
    return [json.loads(r["input_json"]) for r in rows]


def _row(db: Database) -> dict[str, Any]:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM resolutions WHERE run_id = ?", (h.RUN_ID,)).fetchone()
        lineage = conn.execute("SELECT * FROM lineage WHERE run_id = ?", (h.RUN_ID,)).fetchone()
    return {**dict(row), "lineage": dict(lineage)}


def test_agent_is_given_the_guide_not_the_root_cause(tmp_path: Path) -> None:
    db = _run(
        tmp_path, [ConsistencyVerdict(status="pass", problem_state=ProblemState.PENDING_PART)]
    )
    [agent, *_] = _trace_inputs(db, "agent_turn_generator")
    agent_text = json.dumps(agent)
    assert "Wiggle the PSU plug at the back of the unit" in agent_text
    assert "the display flickers" not in agent_text  # a finding only the customer has
    assert "root_cause" not in agent_text and "resolution_hint" not in agent_text
    # The planned outcome is hidden until the agent reports the diagnosis done.
    agents = _trace_inputs(db, "agent_turn_generator")
    assert [a["planned"].get("problem_state") for a in agents] == [None, None, "pending_part"]
    assert agent["planned"] == {"final": True, "revealed": False}

    customer = _trace_inputs(db, "customer_turn_generator")[0]
    findings = customer["diagnosis"]["findings"]
    assert [f["finding"] for f in findings] == [
        "the lamp stays steady",
        "the display flickers when the plug moves",
    ]


def test_planned_outcome_and_commitments_are_stored(tmp_path: Path) -> None:
    db = _run(
        tmp_path, [ConsistencyVerdict(status="pass", problem_state=ProblemState.PENDING_PART)]
    )
    row = _row(db)
    assert row["problem_state"] == row["planned_problem_state"] == "pending_part"
    assert row["contact_ending"] == row["planned_contact_ending"] == "agreed_next_step"
    assert row["resolved"] == 0  # a part on its way is not a fix
    assert json.loads(row["commitments_json"]) == [_PART.model_dump()]
    assert row["quality_flag"] is None
    plan = json.loads(row["lineage"]["case_plan_json"])
    assert row["lineage"]["problem_state"] == "pending_part"
    assert plan["contacts"][0]["checks"] == [0, 1]
    assert plan["contacts"][0]["cause_confirmed"] is True


def test_a_dialogue_that_misses_the_planned_state_is_rerolled(tmp_path: Path) -> None:
    # The checker judges the first attempt as "fixed" although a part is still due.
    db = _run(
        tmp_path,
        [
            ConsistencyVerdict(status="pass", problem_state=ProblemState.FIXED_VERIFIED),
            ConsistencyVerdict(status="pass", problem_state=ProblemState.PENDING_PART),
        ],
    )
    assert h.trace_count(db, node_name="conversation_consistency_check") == 2
    agents = _trace_inputs(db, "agent_turn_generator")
    # On the re-roll the outcome issue stays hidden until the diagnosis is done.
    assert not any("'pending_part'" in i for i in agents[3]["prior_issues"])
    assert any("'pending_part'" in i for i in agents[-1]["prior_issues"])
    assert _row(db)["problem_state"] == "pending_part"


def test_exhausted_retries_keep_the_honest_state(tmp_path: Path) -> None:
    db = _run(
        tmp_path,
        [ConsistencyVerdict(status="pass", problem_state=ProblemState.FIXED_VERIFIED)],
        max_retries=0,
    )
    row = _row(db)
    # What the text reached is stored, next to the plan, and the row is flagged.
    assert row["problem_state"] == "fixed_verified"
    assert row["planned_problem_state"] == "pending_part"
    assert row["quality_flag"] == "warning:retries_exhausted"
