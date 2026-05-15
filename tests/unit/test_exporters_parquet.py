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
            run_id=run_id,
            title="P1",
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
    out = tmp_path / "exports"
    paths = export_run_to_parquet(db, run_id, out_dir=out)
    problems_path = out / run_id / "problems.parquet"
    assert problems_path in paths
    table = pq.read_table(problems_path)
    assert table.num_rows == 1
    assert "title" in table.column_names
