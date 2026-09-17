"""Integration: one case spread over several related calls (rounds) with shared case id."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from csfd.graph.phase2_graph import build_phase2_subgraph
from csfd.models.fake import FakeChatModel
from csfd.pipeline import ConsistencyVerdict, DialogueTurnOutput, IncomingRequestOutput
from csfd.rounds import plan_case_rounds
from csfd.settings import AppSettings, RoundsConfig
from csfd.storage.db import Database
from csfd.storage.transcripts import export_run_transcripts
from tests.integration import _dialogue_harness as h

TICKET_UID = f"{h.RUN_ID}:000001"


def _turn(
    speaker: Literal["customer", "agent"], content: str, done_reason: str | None = None
) -> DialogueTurnOutput:
    return DialogueTurnOutput(
        speaker=speaker,
        content=content,
        done=done_reason is not None,
        done_reason=done_reason,
    )


def _run(
    tmp_path: Path, rounds: RoundsConfig, turns: list[DialogueTurnOutput], *, validation: bool
) -> tuple[Database, AppSettings, FakeChatModel]:
    fake = FakeChatModel(
        structured={ConsistencyVerdict: ConsistencyVerdict(status="pass")},
        structured_seq={
            IncomingRequestOutput: [
                IncomingRequestOutput(subject="Unit power-cycles", body="Hi, it power-cycles."),
                IncomingRequestOutput(
                    subject="Callback: still power-cycling",
                    body="Hi, I called earlier, I reseated it and it still restarts.",
                ),
                IncomingRequestOutput(subject="Third call", body="Hi again, third time now."),
            ],
            DialogueTurnOutput: list(turns),
        },
    )
    db = h.setup_db(tmp_path)
    settings = h.build_settings(
        tmp_path, validation_enabled=validation, max_retries=0, channel="phone", rounds=rounds
    )
    graph = build_phase2_subgraph(factory=h.factory(fake), db=db, settings=settings)
    asyncio.run(graph.ainvoke(h.initial_state(settings)))
    return db, settings, fake


def _follow_up_then_resolved(tmp_path: Path) -> Database:
    rounds = RoundsConfig(proportions={2: 1.0}, callback_reasons={"follow_up": 1.0})
    turns = [
        # call 1: agreed next step, case stays open
        _turn("agent", "Please reseat the power connector and call us back if it recurs."),
        _turn("customer", "Okay, I'll try that and call back.", "follow_up"),
        # call 2: callback, resolved
        _turn("agent", "Thanks for calling back. Let's replace the PSU cable then."),
        _turn("customer", "New cable is in, it's stable now. Thanks!", "customer_satisfied"),
    ]
    db, _, _ = _run(tmp_path, rounds, turns, validation=True)
    return db


def test_follow_up_case_yields_two_linked_calls(tmp_path: Path) -> None:
    db = _follow_up_then_resolved(tmp_path)
    rows = h.resolution_rows(db)
    assert [r["round_index"] for r in rows] == [1, 2]
    assert {r["case_uid"] for r in rows} == {TICKET_UID}
    assert {r["round_count"] for r in rows} == {2}
    assert [r["resolution_uid"] for r in rows] == [f"{TICKET_UID}:res", f"{TICKET_UID}:r02:res"]
    assert [bool(r["resolved"]) for r in rows] == [False, True]
    assert rows[0]["turns"][-1]["done_reason"] == "follow_up"
    assert [r["agent_name"] for r in rows] == [
        "Agent-docs_request-0001",
        "Agent-docs_request-0001-r2",
    ]
    assert "Agent-docs_request-0001-r2" in rows[1]["turns"][0]["content"]
    first_end = datetime.fromisoformat(rows[0]["ended_at"])
    second_start = datetime.fromisoformat(rows[1]["started_at"])
    assert second_start > first_end

    with db.connect() as conn:
        requests = conn.execute(
            "SELECT id, request_uid, case_uid, round_index, subject FROM incoming_requests "
            "ORDER BY round_index"
        ).fetchall()
        lineage = conn.execute("SELECT * FROM lineage").fetchall()
    assert [r["case_uid"] for r in requests] == [TICKET_UID, TICKET_UID]
    assert requests[1]["subject"] == "Callback: still power-cycling"
    # One lineage row per case, linked to the case's first contact (request + its own resolution).
    assert len(lineage) == 1
    assert lineage[0]["incoming_request_id"] == requests[0]["id"]
    assert lineage[0]["resolution_id"] == rows[0]["id"]


def test_callback_prompts_carry_the_earlier_call(tmp_path: Path) -> None:
    db = _follow_up_then_resolved(tmp_path)
    with db.connect() as conn:
        traces = conn.execute(
            "SELECT node_name, artifact_id, input_json FROM agent_traces WHERE run_id = ?",
            (h.RUN_ID,),
        ).fetchall()
    by_contact: dict[str, list[dict[str, object]]] = {}
    for t in traces:
        by_contact.setdefault(t["artifact_id"], []).append(json.loads(t["input_json"]))
    assert set(by_contact) == {TICKET_UID, f"{TICKET_UID}:r02"}
    for inputs in by_contact[TICKET_UID]:
        assert inputs["round"] == {
            "sequence": 1,
            "count": 2,
            "end_mode": "follow_up",
            "since_previous": "some time",
        }
        assert inputs["case_history"] == []
    for inputs in by_contact[f"{TICKET_UID}:r02"]:
        history = inputs["case_history"]
        assert isinstance(history, list) and len(history) == 1
        assert history[0]["ended"] == "follow_up"
        contents = [t["content"] for t in history[0]["turns"]]
        assert "Okay, I'll try that and call back." in contents
        assert inputs["round"]["end_mode"] == "final"  # type: ignore[index]
        assert str(inputs["round"]["since_previous"]).startswith("about ")  # type: ignore[index]


def test_multi_contact_transcript_export_groups_calls(tmp_path: Path) -> None:
    db = _follow_up_then_resolved(tmp_path)
    out = tmp_path / "exports"
    export_run_transcripts(db, h.RUN_ID, out_dir=out)
    case_dir = out / h.RUN_ID / "transcripts" / "case_000001"
    assert sorted(p.name for p in case_dir.iterdir()) == ["call_01.txt", "call_02.txt", "case.json"]
    first = (case_dir / "call_01.txt").read_text()
    second = (case_dir / "call_02.txt").read_text()
    assert "call: 1 of 2" in first
    assert "since_previous_call" not in first
    assert "call: 2 of 2" in second
    assert "since_previous_call: " in second
    assert "agent: Agent-docs_request-0001-r2 (Documentation Rep)" in second

    meta = json.loads((case_dir / "case.json").read_text())
    assert meta["case_id"] == TICKET_UID
    assert meta["contact_count"] == 2
    assert meta["planned_contact_count"] == 2
    assert meta["resolved"] is True
    c1, c2 = meta["contacts"]
    assert (c1["sequence"], c1["outcome"], c1["resolved"]) == (1, "follow_up", False)
    assert (c2["sequence"], c2["file"], c2["resolved"]) == (2, "call_02.txt", True)
    assert c1["gap_since_previous_s"] is None
    assert c2["gap_since_previous_s"] > 3600


def test_dropped_call_is_cut_off_and_called_back(tmp_path: Path) -> None:
    rounds = RoundsConfig(proportions={2: 1.0}, callback_reasons={"dropped": 1.0})
    settings = h.build_settings(tmp_path, validation_enabled=False, channel="phone", rounds=rounds)
    plan = plan_case_rounds(
        slot_index=1,
        round_count=2,
        agent_name="Agent-docs_request-0001",
        rounds=rounds,
        calendar=settings.tickets.calendar,
        seed=settings.pipeline.run_seed or 0,
        opening_turns=2,
        turn_cap=settings.tickets.dialogue.turn_cap,
    )
    drop_after = plan[0].drop_after_turns
    assert drop_after is not None
    # Never-done turns until the line drops, then a short resolved callback.
    call_one = [
        _turn("agent" if i % 2 == 0 else "customer", f"still talking {i}")
        for i in range(drop_after - 2)
    ]
    call_two = [
        _turn("agent", "Sorry we got cut off. Please reseat the connector."),
        _turn("customer", "Done, it works now.", "resolved"),
    ]
    db, _, _ = _run(tmp_path, rounds, call_one + call_two, validation=False)

    first, second = h.resolution_rows(db)
    assert first["turn_count"] == drop_after
    assert first["end_reason"] == "dropped"
    assert not first["resolved"]
    # A planned drop is not a runaway dialogue: no cap warning.
    assert first["quality_flag"] == "warning:validation_skipped"
    assert second["end_reason"] == "customer_done"
    assert second["resolved"]

    out = tmp_path / "exports"
    export_run_transcripts(db, h.RUN_ID, out_dir=out)
    text = (out / h.RUN_ID / "transcripts" / "case_000001" / "call_01.txt").read_text()
    assert text.rstrip().endswith("(call disconnected)")
    no_ts = tmp_path / "exports_plain"
    export_run_transcripts(db, h.RUN_ID, out_dir=no_ts, timestamps=False)
    plain = (no_ts / h.RUN_ID / "transcripts" / "case_000001" / "call_01.txt").read_text()
    assert plain.rstrip().splitlines()[-1] == "(call disconnected)"


def test_single_contact_default_keeps_trace_inputs_unchanged(tmp_path: Path) -> None:
    rounds = RoundsConfig()
    turns = [_turn("agent", "Reseat it."), _turn("customer", "Fixed, thanks!", "resolved")]
    db, _, _ = _run(tmp_path, rounds, turns, validation=True)
    with db.connect() as conn:
        inputs = [
            json.loads(r["input_json"])
            for r in conn.execute("SELECT input_json FROM agent_traces").fetchall()
        ]
    assert inputs
    assert all("round" not in i and "case_history" not in i for i in inputs)
    (row,) = h.resolution_rows(db)
    assert (row["case_uid"], row["round_index"], row["round_count"]) == (TICKET_UID, 1, 1)
