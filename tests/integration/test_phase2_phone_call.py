"""Integration: the phone channel produces a timed call transcript and exports it as text."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from csfd.agents.factory import AgentFactory
from csfd.graph.phase2_graph import build_phase2_subgraph
from csfd.models.fake import FakeChatModel
from csfd.pipeline import (
    ConsistencyVerdict,
    DialogueTurnOutput,
    IncomingRequestOutput,
)
from csfd.prompts.registry import PromptRegistry
from csfd.storage.db import Database
from csfd.storage.transcripts import export_run_transcripts
from tests.integration import _dialogue_harness as h


def _factory() -> AgentFactory:
    fake = FakeChatModel(
        structured={
            IncomingRequestOutput: IncomingRequestOutput(
                subject="Unit power-cycles every 20 minutes",
                body="Hi, um, yeah, our unit keeps power-cycling about every twenty minutes.",
            ),
            ConsistencyVerdict: ConsistencyVerdict(status="pass"),
        },
        structured_seq={
            DialogueTurnOutput: [
                DialogueTurnOutput(
                    speaker="agent",
                    content="Okay, can you hold for a moment while I check our notes?",
                ),
                DialogueTurnOutput(speaker="customer", content="Sure, go ahead."),
                DialogueTurnOutput(
                    speaker="agent",
                    content="[hold] Thanks for holding. Please reseat the power connector.",
                ),
                DialogueTurnOutput(
                    speaker="customer",
                    content="Okay [pause] done, it's stable now. Thanks!",
                    done=True,
                    done_reason="customer_satisfied",
                ),
            ],
        },
    )
    return h.factory(fake)


def _run_phone_slot(tmp_path: Path) -> Database:
    db = h.setup_db(tmp_path)
    settings = h.build_settings(tmp_path, validation_enabled=True, max_retries=0, channel="phone")
    graph = build_phase2_subgraph(factory=_factory(), db=db, settings=settings)
    asyncio.run(graph.ainvoke(h.initial_state(settings)))
    return db


def test_phone_call_starts_with_scripted_greeting_and_is_timed(tmp_path: Path) -> None:
    db = _run_phone_slot(tmp_path)
    row = h.resolution_row(db)

    turns = row["turns"]
    # greeting (scripted) + caller opening + 4 scripted turns
    assert row["turn_count"] == 6
    assert [t["speaker"] for t in turns[:2]] == ["agent", "customer"]
    assert "Acme" in turns[0]["content"]
    assert "Agent-docs_request-0001" in turns[0]["content"]
    assert turns[1]["content"].startswith("Hi, um")
    assert row["resolved"] is True
    assert row["end_reason"] == "customer_done"

    # Per-utterance offsets: ordered, non-overlapping here, hold gap recorded.
    starts = [t["start_s"] for t in turns]
    assert starts == sorted(starts)
    assert starts[0] == 0.0
    assert all(t["end_s"] > t["start_s"] for t in turns)
    assert turns[4]["hold_s"] >= 30.0
    assert "hold_s" not in turns[3]

    assert row["channel"] == "phone"
    assert row["agent_name"] == "Agent-docs_request-0001"
    assert row["duration_s"] == turns[-1]["end_s"]
    started = datetime.fromisoformat(row["started_at"])
    ended = datetime.fromisoformat(row["ended_at"])
    assert round((ended - started).total_seconds(), 1) == row["duration_s"]
    assert started.weekday() < 5

    with db.connect() as conn:
        channel = conn.execute("SELECT channel, subject FROM incoming_requests").fetchone()
    assert channel["channel"] == "phone"
    assert channel["subject"] == "Unit power-cycles every 20 minutes"


def test_phone_call_uses_phone_prompts(tmp_path: Path) -> None:
    db = _run_phone_slot(tmp_path)
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT node_name, prompt_id FROM agent_traces WHERE run_id = ?", (h.RUN_ID,)
        ).fetchall()
    prompt_ids = {(r["node_name"], r["prompt_id"]) for r in rows}
    assert ("incoming_request_generator", reg.get("phase2.phone_incoming_request").version) in (
        prompt_ids
    )
    assert ("agent_turn_generator", reg.get("phase2.phone_agent_turn").version) in prompt_ids
    assert ("customer_turn_generator", reg.get("phase2.phone_customer_turn").version) in (
        prompt_ids
    )


def test_phone_transcript_export(tmp_path: Path) -> None:
    db = _run_phone_slot(tmp_path)
    out = tmp_path / "exports"
    export_run_transcripts(db, h.RUN_ID, out_dir=out)

    case_dir = out / h.RUN_ID / "transcripts" / "case_000001"
    text = (case_dir / "call_01.txt").read_text(encoding="utf-8")
    header, _, body = text.partition("\n\n")
    assert header.splitlines()[0] == "CALL TRANSCRIPT"
    assert "call: 1 of 1" in header
    assert "channel: phone (inbound)" in header
    assert re.search(r"^duration: \d{2}:\d{2}:\d{2}$", header, re.MULTILINE)
    assert "caller: Customer-standard-0001 (standard tier)" in header
    assert "agent: Agent-docs_request-0001 (Documentation Rep)" in header
    # Ground truth never reaches the transcript folder.
    assert "loose PSU connector" not in text

    lines = body.splitlines()
    assert lines[0].startswith("[00:00:00] AGENT: ")
    assert lines[1].startswith("[00:00:") and "CUSTOMER: Hi, um" in lines[1]
    hold_line = next(line for line in lines if "caller on hold" in line)
    resumed = lines[lines.index(hold_line) + 1]
    assert re.match(r"^\[\d{2}:\d{2}:\d{2}\] AGENT: Thanks for holding\.", resumed)
    assert "[hold]" not in body

    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert meta["channel"] == "phone"
    assert meta["contact_count"] == 1
    assert meta["resolved"] is True
    assert meta["contacts"][0]["file"] == "call_01.txt"
    assert meta["contacts"][0]["gap_since_previous_s"] is None
    assert "utterances" not in meta["contacts"][0]

    index = (out / h.RUN_ID / "transcripts" / "cases.jsonl").read_text().splitlines()
    assert len(index) == 1
    utterances = json.loads(index[0])["contacts"][0]["utterances"]
    assert len(utterances) == 6
    assert {"speaker", "text", "start_s", "end_s"} <= set(utterances[0])


def test_phone_transcript_export_without_timestamps(tmp_path: Path) -> None:
    db = _run_phone_slot(tmp_path)
    out = tmp_path / "exports"
    export_run_transcripts(db, h.RUN_ID, out_dir=out, timestamps=False)
    text = (out / h.RUN_ID / "transcripts" / "case_000001" / "call_01.txt").read_text()
    body = text.partition("\n\n")[2]
    assert body.splitlines()[0].startswith("AGENT: ")
    assert not re.search(r"^\[\d{2}:", body, re.MULTILINE)
    assert "(caller on hold, " in body


def test_phone_transcript_header_times_are_whole_seconds(tmp_path: Path) -> None:
    db = _run_phone_slot(tmp_path)
    out = tmp_path / "exports"
    export_run_transcripts(db, h.RUN_ID, out_dir=out)
    header = (
        (out / h.RUN_ID / "transcripts" / "case_000001" / "call_01.txt")
        .read_text()
        .partition("\n\n")[0]
    )
    for key in ("started_at", "ended_at"):
        (value,) = [line.split(": ", 1)[1] for line in header.splitlines() if line.startswith(key)]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", value), value
