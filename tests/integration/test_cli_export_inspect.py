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
from csfd.storage.repository import RunRecord, RunRepo
from csfd.storage.v2_repository import ProblemV2Record, ProblemV2Repo

runner = CliRunner()


def _seed_run(db_path: Path) -> str:
    db = Database(path=db_path)
    apply_migrations(db)
    rid = str(uuid4())
    RunRepo(db).create(
        RunRecord(
            id=rid,
            phase="full",
            parent_run_id=None,
            status="completed",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            run_seed=1,
            pipeline_version="0.3.0",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    ProblemV2Repo(db).create(
        ProblemV2Record(
            id=f"{rid}:p:0000",
            run_id=rid,
            title="t",
            summary="s",
            background="b",
            category="c",
            complexity="simple",
            resolution_hints={"l1": "step"},
            quality_flag=None,
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
    assert "problem" in result.output.lower()
