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
