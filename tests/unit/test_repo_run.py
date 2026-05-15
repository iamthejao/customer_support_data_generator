import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import RunRecord, RunRepo


def _setup(tmp_db_path: Path) -> RunRepo:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    return RunRepo(db)


def test_create_and_get_run(tmp_db_path: Path) -> None:
    repo = _setup(tmp_db_path)
    run = RunRecord(
        id=str(uuid4()),
        phase="phase1",
        parent_run_id=None,
        status="running",
        started_at=datetime.now(timezone.utc),  # noqa: UP017
        completed_at=None,
        run_seed=42,
        pipeline_version="0.1.0",
        git_sha="abc1234",
        config_snapshot_json=json.dumps({"k": "v"}),
        stats_json=None,
        error_summary=None,
    )
    repo.create(run)
    got = repo.get(run.id)
    assert got.id == run.id
    assert got.run_seed == 42
    assert got.status == "running"


def test_update_status_and_stats(tmp_db_path: Path) -> None:
    repo = _setup(tmp_db_path)
    run_id = str(uuid4())
    repo.create(
        RunRecord(
            id=run_id,
            phase="phase2",
            parent_run_id=None,
            status="running",
            started_at=datetime.now(timezone.utc),  # noqa: UP017
            completed_at=None,
            run_seed=1,
            pipeline_version="0.1.0",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    repo.update_status(run_id, status="completed", stats={"tokens": 123})
    got = repo.get(run_id)
    assert got.status == "completed"
    assert got.stats_json is not None
    assert json.loads(got.stats_json)["tokens"] == 123


def test_get_unknown_id_raises(tmp_db_path: Path) -> None:
    repo = _setup(tmp_db_path)
    with pytest.raises(KeyError):
        repo.get("does-not-exist")
