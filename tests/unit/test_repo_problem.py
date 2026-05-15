import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
)


def _setup_run(tmp_db_path: Path) -> tuple[Database, str]:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="running",
        started_at=datetime.now(UTC), completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None, config_snapshot_json="{}",
        stats_json=None, error_summary=None,
    ))
    return db, run_id


def test_create_and_list_problems(tmp_db_path: Path) -> None:
    db, run_id = _setup_run(tmp_db_path)
    repo = ProblemRepo(db)
    for i in range(3):
        repo.create(ProblemRecord(
            id=str(uuid4()), run_id=run_id,
            title=f"Issue {i}", description="desc",
            category="billing", severity="medium",
            has_kb=(i % 2 == 0), coverage_reasoning="r",
            coverage_confidence="medium",
            metadata_json=json.dumps({"tag": i}),
            quality_flag=None, unresolved_issues_json=None,
            created_at=datetime.now(UTC),
        ))
    rows = repo.list_for_run(run_id)
    assert len(rows) == 3
    has_kb = repo.list_for_run(run_id, has_kb=True)
    assert len(has_kb) == 2


def test_create_is_idempotent_on_duplicate_id(tmp_db_path: Path) -> None:
    db, run_id = _setup_run(tmp_db_path)
    repo = ProblemRepo(db)
    pid = str(uuid4())
    rec = ProblemRecord(
        id=pid, run_id=run_id, title="t", description="d",
        category="c", severity="low", has_kb=False,
        coverage_reasoning=None, coverage_confidence=None,
        metadata_json=None, quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(UTC),
    )
    repo.create(rec)
    repo.create(rec)
    assert len(repo.list_for_run(run_id)) == 1
