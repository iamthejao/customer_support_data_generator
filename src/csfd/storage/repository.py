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


@dataclass(slots=True)
class ProblemRecord:
    id: str
    run_id: str
    title: str
    description: str
    category: str
    severity: str
    has_kb: bool
    coverage_reasoning: str | None
    coverage_confidence: str | None
    metadata_json: str | None
    quality_flag: str | None
    unresolved_issues_json: str | None
    created_at: datetime


class ProblemRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, p: ProblemRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO problems
                  (id, run_id, title, description, category, severity, has_kb,
                   coverage_reasoning, coverage_confidence, metadata_json,
                   quality_flag, unresolved_issues_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p.id, p.run_id, p.title, p.description, p.category, p.severity,
                    1 if p.has_kb else 0,
                    p.coverage_reasoning, p.coverage_confidence, p.metadata_json,
                    p.quality_flag, p.unresolved_issues_json,
                    p.created_at.isoformat(),
                ),
            )

    def list_for_run(self, run_id: str, *, has_kb: bool | None = None) -> list[ProblemRecord]:
        sql = "SELECT * FROM problems WHERE run_id = ?"
        params: list[Any] = [run_id]
        if has_kb is not None:
            sql += " AND has_kb = ?"
            params.append(1 if has_kb else 0)
        sql += " ORDER BY created_at"
        with self.db.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            ProblemRecord(
                id=r["id"], run_id=r["run_id"], title=r["title"],
                description=r["description"], category=r["category"],
                severity=r["severity"], has_kb=bool(r["has_kb"]),
                coverage_reasoning=r["coverage_reasoning"],
                coverage_confidence=r["coverage_confidence"],
                metadata_json=r["metadata_json"],
                quality_flag=r["quality_flag"],
                unresolved_issues_json=r["unresolved_issues_json"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]


@dataclass(slots=True)
class KBArticleRecord:
    id: str
    run_id: str
    problem_id: str
    title: str
    content_markdown: str
    content_hash: str
    troubleshooting_steps_json: str
    prerequisites_json: str | None
    metadata_json: str | None
    version: int
    quality_flag: str | None
    unresolved_issues_json: str | None
    created_at: datetime


class KBArticleRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, a: KBArticleRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO kb_articles
                  (id, run_id, problem_id, title, content_markdown, content_hash,
                   troubleshooting_steps_json, prerequisites_json, metadata_json,
                   version, quality_flag, unresolved_issues_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a.id, a.run_id, a.problem_id, a.title, a.content_markdown,
                    a.content_hash, a.troubleshooting_steps_json,
                    a.prerequisites_json, a.metadata_json, a.version,
                    a.quality_flag, a.unresolved_issues_json,
                    a.created_at.isoformat(),
                ),
            )

    def get_by_problem(self, problem_id: str) -> KBArticleRecord | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM kb_articles WHERE problem_id = ?", (problem_id,)
            ).fetchone()
        if row is None:
            return None
        return KBArticleRecord(
            id=row["id"], run_id=row["run_id"], problem_id=row["problem_id"],
            title=row["title"], content_markdown=row["content_markdown"],
            content_hash=row["content_hash"],
            troubleshooting_steps_json=row["troubleshooting_steps_json"],
            prerequisites_json=row["prerequisites_json"],
            metadata_json=row["metadata_json"],
            version=row["version"],
            quality_flag=row["quality_flag"],
            unresolved_issues_json=row["unresolved_issues_json"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
