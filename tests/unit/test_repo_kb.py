import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRecord,
    KBArticleRepo,
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
)


def _make_problem(db: Database) -> tuple[str, str]:
    run_id = str(uuid4())
    RunRepo(db).create(
        RunRecord(
            id=run_id,
            phase="phase1",
            parent_run_id=None,
            status="running",
            started_at=datetime.now(UTC),
            completed_at=None,
            run_seed=1,
            pipeline_version="0.1.0",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    pid = str(uuid4())
    ProblemRepo(db).create(
        ProblemRecord(
            id=pid,
            run_id=run_id,
            title="t",
            description="d",
            category="c",
            severity="low",
            has_kb=True,
            coverage_reasoning=None,
            coverage_confidence=None,
            metadata_json=None,
            quality_flag=None,
            unresolved_issues_json=None,
            created_at=datetime.now(UTC),
        )
    )
    return run_id, pid


def test_create_and_get_kb_article(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id, pid = _make_problem(db)

    repo = KBArticleRepo(db)
    aid = str(uuid4())
    repo.create(
        KBArticleRecord(
            id=aid,
            run_id=run_id,
            problem_id=pid,
            title="How to reset",
            content_markdown="## Step 1\n...",
            content_hash="abcdef012345",
            troubleshooting_steps_json=json.dumps([{"step": "do X", "expected_result": "Y"}]),
            prerequisites_json=None,
            metadata_json=None,
            version=1,
            quality_flag=None,
            unresolved_issues_json=None,
            created_at=datetime.now(UTC),
        )
    )
    got = repo.get_by_problem(pid)
    assert got is not None
    assert got.id == aid
    assert got.title == "How to reset"


def test_get_by_problem_returns_none_when_absent(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    _, pid = _make_problem(db)
    repo = KBArticleRepo(db)
    assert repo.get_by_problem(pid) is None
