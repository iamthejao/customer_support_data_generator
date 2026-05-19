import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.storage.db import Database
from csfd.storage.exporters import export_run_to_jsonl
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import ProblemRecord, ProblemRepo, RunRecord, RunRepo


def test_export_jsonl_writes_problems_file(tmp_path: Path) -> None:
    db_path = tmp_path / "x.sqlite"
    db = Database(path=db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(
        RunRecord(
            id=run_id,
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
    repo = ProblemRepo(db)
    for i in range(2):
        repo.create(
            ProblemRecord(
                id=f"{run_id}:p:{i:04d}",
                run_id=run_id,
                title=f"P{i}",
                summary="s",
                background="b",
                category="c",
                complexity="simple",
                resolution_hints={"l1": "step"},
                quality_flag=None,
                created_at=datetime.now(UTC),
            )
        )
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


def test_jsonl_export_includes_problem_embeddings(tmp_path: Path) -> None:
    import asyncio
    from datetime import UTC, datetime

    from csfd.storage.db_async import AsyncDatabase
    from csfd.storage.repository import (
        ProblemEmbeddingRecord,
        ProblemEmbeddingRepo,
        ProblemRecord,
        ProblemRepo,
        RunRecord,
        RunRepo,
    )

    db = Database(str(tmp_path / "t.sqlite"))
    apply_migrations(db)
    now = datetime.now(UTC)
    RunRepo(db).create(
        RunRecord(
            id="r",
            phase="full",
            parent_run_id=None,
            status="completed",
            started_at=now,
            completed_at=now,
            run_seed=1,
            pipeline_version="t",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json="{}",
            error_summary=None,
        )
    )
    ProblemRepo(db).create(
        ProblemRecord(
            id="r:p:0000",
            run_id="r",
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
    asyncio.run(
        ProblemEmbeddingRepo(db).acreate(
            AsyncDatabase(db.path),
            ProblemEmbeddingRecord(
                problem_id="r:p:0000",
                run_id="r",
                model="m",
                dim=3,
                vector=(0.1, 0.2, 0.3),
                created_at=now,
            ),
        )
    )

    export_run_to_jsonl(db, "r", out_dir=tmp_path / "out")
    f = tmp_path / "out" / "r" / "problem_embeddings.jsonl"
    assert f.exists()
    rows = [json.loads(line) for line in f.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["problem_id"] == "r:p:0000"
    assert rows[0]["dim"] == 3
    assert rows[0]["model"] == "m"
    assert isinstance(rows[0]["vector"], list)
    assert len(rows[0]["vector"]) == 3
    assert rows[0]["vector"][0] == pytest.approx(0.1, rel=1e-5)
    assert rows[0]["vector"][1] == pytest.approx(0.2, rel=1e-5)
    assert rows[0]["vector"][2] == pytest.approx(0.3, rel=1e-5)

    manifest = json.loads((tmp_path / "out" / "r" / "manifest.json").read_text())
    assert "problem_embeddings.jsonl" in manifest["files"]
