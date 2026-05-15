import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.exporters import export_run_to_jsonl
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
)


def test_export_jsonl_writes_problems_file(tmp_path: Path) -> None:
    db_path = tmp_path / "x.sqlite"
    db = Database(path=db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="completed",
        started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        run_seed=1, pipeline_version="0.1.0", git_sha=None,
        config_snapshot_json="{}", stats_json=None, error_summary=None,
    ))
    repo = ProblemRepo(db)
    for i in range(2):
        repo.create(ProblemRecord(
            id=str(uuid4()), run_id=run_id, title=f"P{i}",
            description="d", category="c", severity="low",
            has_kb=False, coverage_reasoning=None, coverage_confidence=None,
            metadata_json=None, quality_flag=None,
            unresolved_issues_json=None,
            created_at=datetime.now(UTC),
        ))
    out = tmp_path / "exports"
    paths = export_run_to_jsonl(db, run_id, out_dir=out)
    problems_path = out / run_id / "problems.jsonl"
    assert problems_path in paths
    lines = problems_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["run_id"] == run_id
    manifest = json.loads((out / run_id / "manifest.json").read_text())
    assert "problems.jsonl" in manifest["files"]
