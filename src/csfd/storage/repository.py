"""Data-access layer. Each *Repo encapsulates idempotent CRUD for one table."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from csfd.storage.db import Database


@dataclass(slots=True)
class RunRecord:
    id: str
    phase: str
    parent_run_id: str | None
    status: str
    started_at: datetime
    completed_at: datetime | None
    run_seed: int
    pipeline_version: str
    git_sha: str | None
    config_snapshot_json: str
    stats_json: str | None
    error_summary: str | None


class RunRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, run: RunRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO runs
                  (id, phase, parent_run_id, status, started_at, completed_at,
                   run_seed, pipeline_version, git_sha, config_snapshot_json,
                   stats_json, error_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id, run.phase, run.parent_run_id, run.status,
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                    run.run_seed, run.pipeline_version, run.git_sha,
                    run.config_snapshot_json, run.stats_json, run.error_summary,
                ),
            )

    def get(self, run_id: str) -> RunRecord:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return _row_to_run(row)

    def update_status(
        self,
        run_id: str,
        *,
        status: str,
        stats: dict[str, Any] | None = None,
        error_summary: str | None = None,
        completed: bool = True,
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, completed_at = ?, stats_json = COALESCE(?, stats_json),
                    error_summary = COALESCE(?, error_summary)
                WHERE id = ?
                """,
                (
                    status,
                    datetime.now(timezone.utc).isoformat() if completed else None,  # noqa: UP017
                    json.dumps(stats) if stats is not None else None,
                    error_summary,
                    run_id,
                ),
            )


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        phase=row["phase"],
        parent_run_id=row["parent_run_id"],
        status=row["status"],
        started_at=datetime.fromisoformat(row["started_at"]),
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        run_seed=row["run_seed"],
        pipeline_version=row["pipeline_version"],
        git_sha=row["git_sha"],
        config_snapshot_json=row["config_snapshot_json"],
        stats_json=row["stats_json"],
        error_summary=row["error_summary"],
    )
