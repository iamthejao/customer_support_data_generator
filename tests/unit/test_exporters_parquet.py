from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from csfd.storage.db import Database
from csfd.storage.exporters import export_run_to_parquet
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import ProblemRecord, ProblemRepo, RunRecord, RunRepo


def test_export_parquet_round_trip(tmp_path: Path) -> None:
    db = Database(path=tmp_path / "x.sqlite")
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
    ProblemRepo(db).create(
        ProblemRecord(
            id=f"{run_id}:p:0000",
            run_id=run_id,
            title="P1",
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
    paths = export_run_to_parquet(db, run_id, out_dir=out)
    problems_path = out / run_id / "problems.parquet"
    assert problems_path in paths
    table = pq.read_table(problems_path)
    assert table.num_rows == 1
    assert "title" in table.column_names


def test_parquet_export_includes_problem_embeddings(tmp_path: Path) -> None:
    import asyncio
    from datetime import UTC, datetime

    import pyarrow.parquet as pq

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

    export_run_to_parquet(db, "r", out_dir=tmp_path / "out")
    f = tmp_path / "out" / "r" / "problem_embeddings.parquet"
    assert f.exists()
    table = pq.read_table(f)
    assert table.num_rows == 1
    cols = {field.name for field in table.schema}
    assert {"problem_id", "run_id", "model", "dim", "vector", "created_at"}.issubset(cols)
    vec = table.column("vector")[0].as_py()
    assert isinstance(vec, list)
    assert len(vec) == 3
