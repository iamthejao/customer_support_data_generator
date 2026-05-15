"""CLI export and inspect command tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from csfd.cli import app
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
)

runner = CliRunner()


def _seed_run(db_path: Path) -> str:
    db = Database(path=db_path)
    apply_migrations(db)
    rid = str(uuid4())
    RunRepo(db).create(
        RunRecord(
            id=rid,
            phase="phase1",
            parent_run_id=None,
            status="completed",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            run_seed=1,
            pipeline_version="0.1.0",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    ProblemRepo(db).create(
        ProblemRecord(
            id=str(uuid4()),
            run_id=rid,
            title="t",
            description="d",
            category="c",
            severity="low",
            has_kb=True,
            coverage_reasoning="r",
            coverage_confidence="high",
            metadata_json=None,
            quality_flag=None,
            unresolved_issues_json=None,
            created_at=datetime.now(UTC),
        )
    )
    return rid


def test_export_writes_jsonl(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.sqlite"
    rid = _seed_run(db_path)
    out = tmp_path / "exports"
    result = runner.invoke(
        app,
        [
            "export",
            rid,
            "--format",
            "jsonl",
            "--sqlite-path",
            str(db_path),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    files = list((out / rid).glob("*.jsonl"))
    names = {f.name for f in files}
    assert "problems.jsonl" in names
    body = (out / rid / "problems.jsonl").read_text().strip()
    parsed = json.loads(body.splitlines()[0])
    assert parsed["title"] == "t"


def test_inspect_prints_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.sqlite"
    rid = _seed_run(db_path)
    result = runner.invoke(
        app,
        [
            "inspect",
            rid,
            "--sqlite-path",
            str(db_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert rid in result.output
    assert "problems" in result.output.lower()
