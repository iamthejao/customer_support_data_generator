"""CLI: `csfd export --format transcripts` and the phone-related `generate` options."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from csfd.cli import app
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    IncomingRequestRecord,
    IncomingRequestRepo,
    LineageRecord,
    LineageRepo,
    ProblemRecord,
    ProblemRepo,
    ResolutionRecord,
    ResolutionRepo,
    RunRecord,
    RunRepo,
)

runner = CliRunner()
RUN_ID = "run-email"


def _seed_email_case(db_path: Path) -> None:
    db = Database(path=db_path)
    apply_migrations(db)
    now = datetime.now(UTC)
    RunRepo(db).create(
        RunRecord(
            id=RUN_ID,
            phase="full",
            parent_run_id=None,
            status="completed",
            started_at=now,
            completed_at=now,
            run_seed=1,
            pipeline_version="test",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    ProblemRepo(db).create(
        ProblemRecord(
            id=f"{RUN_ID}:p:0000",
            run_id=RUN_ID,
            title="t",
            summary="s",
            background="b",
            category="c",
            complexity="simple",
            resolution_hints={},
            quality_flag=None,
            created_at=now,
            root_cause=["SECRET ROOT CAUSE"],
        )
    )
    ticket_uid = f"{RUN_ID}:000001"
    LineageRepo(db).create(
        LineageRecord(
            ticket_uid=ticket_uid,
            run_id=RUN_ID,
            slot_index=1,
            problem_id=f"{RUN_ID}:p:0000",
            ticket_type="l1",
            customer_tier="premium",
            customer_tone="polite",
            incoming_request_id=None,
            resolution_id=None,
            created_at=now,
        )
    )
    ir_id = IncomingRequestRepo(db).create(
        IncomingRequestRecord(
            request_uid=f"{ticket_uid}:req",
            run_id=RUN_ID,
            problem_id=f"{RUN_ID}:p:0000",
            ticket_type="l1",
            customer_name="Customer-premium-0001",
            customer_tier="premium",
            customer_tone="polite",
            channel="email",
            subject="Display flickers",
            body="Hello, the display flickers.",
            quality_flag=None,
            created_at=now,
        )
    )
    res_id = ResolutionRepo(db).create(
        ResolutionRecord(
            resolution_uid=f"{ticket_uid}:res",
            run_id=RUN_ID,
            incoming_request_id=ir_id,
            problem_id=f"{RUN_ID}:p:0000",
            ticket_type="l1",
            turns=[
                {"speaker": "customer", "content": "Hello, the display flickers."},
                {
                    "speaker": "agent",
                    "content": "Please update the firmware.",
                    "done": True,
                    "done_reason": "resolved",
                },
            ],
            turn_count=2,
            resolved=True,
            quality_flag=None,
            created_at=now,
            agent_name="Agent-l1-0001",
            end_reason="agent_done",
            started_at=datetime(2026, 1, 6, 9, 15, tzinfo=UTC),
        )
    )
    LineageRepo(db).update_links(ticket_uid, incoming_request_id=ir_id, resolution_id=res_id)


def test_export_transcripts_writes_email_exchange(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.sqlite"
    _seed_email_case(db_path)
    out = tmp_path / "exports"
    args = ["export", RUN_ID, "--format", "transcripts"]
    result = runner.invoke(app, [*args, "--sqlite-path", str(db_path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    root = out / RUN_ID / "transcripts"
    text = (root / "case_000001" / "email_01.txt").read_text(encoding="utf-8")
    assert text.startswith("EMAIL EXCHANGE\ncase_id: run-email:000001\nemail: 1 of 1\n")
    assert "started_at: 2026-01-06T09:15:00+00:00" in text
    assert "subject: Display flickers" in text
    assert "customer: Customer-premium-0001 (premium tier)" in text
    assert "--- CUSTOMER ---\nHello, the display flickers.\n\n--- AGENT ---\n" in text
    assert "duration:" not in text
    assert "SECRET ROOT CAUSE" not in text
    meta = json.loads((root / "case_000001" / "case.json").read_text())
    assert meta["contacts"][0]["outcome"] == "resolved"
    assert meta["contacts"][0]["end_reason"] == "agent_done"
    # Only the transcripts were requested.
    assert not (out / RUN_ID / "problems.jsonl").exists()


def test_export_all_writes_tables_and_transcripts(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.sqlite"
    _seed_email_case(db_path)
    out = tmp_path / "exports"
    result = runner.invoke(
        app, ["export", RUN_ID, "--format", "all", "--sqlite-path", str(db_path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (out / RUN_ID / "resolutions.jsonl").exists()
    assert (out / RUN_ID / "resolutions.parquet").exists()
    assert (out / RUN_ID / "transcripts" / "cases.jsonl").exists()


def test_export_rejects_unknown_format() -> None:
    result = runner.invoke(app, ["export", RUN_ID, "--format", "pdf"])
    assert result.exit_code != 0
    assert "must be one of" in result.output


def test_generate_rejects_unknown_channel() -> None:
    result = runner.invoke(app, ["generate", "--channel", "fax"])
    assert result.exit_code != 0
    assert "must be one of: email, phone" in result.output


def test_generate_rejects_unknown_disfluency() -> None:
    result = runner.invoke(app, ["generate", "--disfluency", "heavy"])
    assert result.exit_code != 0
    assert "must be one of: none, light, moderate" in result.output
