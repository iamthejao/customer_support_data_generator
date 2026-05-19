"""ProblemEmbeddingRepo CRUD: sync list, async create, idempotent re-create."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from csfd.storage.db import Database
from csfd.storage.db_async import AsyncDatabase
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemEmbeddingRecord,
    ProblemEmbeddingRepo,
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
)


def _seed_run_and_problem(db: Database, *, run_id: str, problem_id: str) -> None:
    now = datetime.now(UTC)
    RunRepo(db).create(
        RunRecord(
            id=run_id,
            phase="full",
            parent_run_id=None,
            status="running",
            started_at=now,
            completed_at=None,
            run_seed=1,
            pipeline_version="t",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    ProblemRepo(db).create(
        ProblemRecord(
            id=problem_id,
            run_id=run_id,
            title="t",
            summary="s",
            background="b",
            category="c",
            complexity="simple",
            resolution_hints={"docs_request": "x", "l1": "x", "l2": "x", "l3": "x"},
            quality_flag=None,
            created_at=now,
        )
    )


def test_acreate_and_list_for_run_roundtrip(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "t.sqlite"))
    apply_migrations(db)
    _seed_run_and_problem(db, run_id="r1", problem_id="r1:p:0000")
    adb = AsyncDatabase(db.path)
    rec = ProblemEmbeddingRecord(
        problem_id="r1:p:0000",
        run_id="r1",
        model="embeddinggemma:300m",
        dim=3,
        vector=(1.0, 0.0, 0.0),
        created_at=datetime.now(UTC),
    )

    asyncio.run(ProblemEmbeddingRepo(db).acreate(adb, rec))

    rows = ProblemEmbeddingRepo(db).list_for_run("r1")
    assert len(rows) == 1
    got = rows[0]
    assert got.problem_id == "r1:p:0000"
    assert got.dim == 3
    assert got.vector == (1.0, 0.0, 0.0)
    assert got.model == "embeddinggemma:300m"


def test_acreate_is_idempotent_on_pk_collision(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "t.sqlite"))
    apply_migrations(db)
    _seed_run_and_problem(db, run_id="r1", problem_id="r1:p:0000")
    adb = AsyncDatabase(db.path)
    rec = ProblemEmbeddingRecord(
        problem_id="r1:p:0000",
        run_id="r1",
        model="m",
        dim=2,
        vector=(1.0, 0.0),
        created_at=datetime.now(UTC),
    )
    asyncio.run(ProblemEmbeddingRepo(db).acreate(adb, rec))
    # Second call must not raise and must not insert a duplicate.
    asyncio.run(ProblemEmbeddingRepo(db).acreate(adb, rec))
    rows = ProblemEmbeddingRepo(db).list_for_run("r1")
    assert len(rows) == 1


def test_acreate_rejects_dim_vector_mismatch(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "t.sqlite"))
    apply_migrations(db)
    _seed_run_and_problem(db, run_id="r1", problem_id="r1:p:0000")
    adb = AsyncDatabase(db.path)
    rec = ProblemEmbeddingRecord(
        problem_id="r1:p:0000",
        run_id="r1",
        model="m",
        dim=3,
        vector=(1.0, 0.0),  # mismatched
        created_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="vector length"):
        asyncio.run(ProblemEmbeddingRepo(db).acreate(adb, rec))
