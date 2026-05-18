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
                    run.id,
                    run.phase,
                    run.parent_run_id,
                    run.status,
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                    run.run_seed,
                    run.pipeline_version,
                    run.git_sha,
                    run.config_snapshot_json,
                    run.stats_json,
                    run.error_summary,
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
class AgentTraceRecord:
    id: str
    run_id: str
    thread_id: str
    node_name: str
    agent_role: str
    artifact_type: str
    artifact_id: str | None
    attempt: int
    prompt_id: str
    input_json: str
    output_json: str | None
    verdict: str | None
    verdict_issues_json: str | None
    model_provider: str
    model_id: str
    tokens_in: int | None
    tokens_out: int | None
    cost_usd_estimated: float | None
    latency_ms: int | None
    parent_trace_id: str | None
    status: str
    error_class: str | None
    error_message: str | None
    created_at: datetime


class AgentTraceRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, t: AgentTraceRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_traces
                  (id, run_id, thread_id, node_name, agent_role,
                   artifact_type, artifact_id, attempt, prompt_id,
                   input_json, output_json, verdict, verdict_issues_json,
                   model_provider, model_id, tokens_in, tokens_out,
                   cost_usd_estimated, latency_ms, parent_trace_id,
                   status, error_class, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t.id,
                    t.run_id,
                    t.thread_id,
                    t.node_name,
                    t.agent_role,
                    t.artifact_type,
                    t.artifact_id,
                    t.attempt,
                    t.prompt_id,
                    t.input_json,
                    t.output_json,
                    t.verdict,
                    t.verdict_issues_json,
                    t.model_provider,
                    t.model_id,
                    t.tokens_in,
                    t.tokens_out,
                    t.cost_usd_estimated,
                    t.latency_ms,
                    t.parent_trace_id,
                    t.status,
                    t.error_class,
                    t.error_message,
                    t.created_at.isoformat(),
                ),
            )

    def list_for_run(self, run_id: str, *, agent_role: str | None = None) -> list[AgentTraceRecord]:
        sql = "SELECT * FROM agent_traces WHERE run_id = ?"
        params: list[Any] = [run_id]
        if agent_role:
            sql += " AND agent_role = ?"
            params.append(agent_role)
        sql += " ORDER BY created_at"
        with self.db.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            AgentTraceRecord(
                id=r["id"],
                run_id=r["run_id"],
                thread_id=r["thread_id"],
                node_name=r["node_name"],
                agent_role=r["agent_role"],
                artifact_type=r["artifact_type"],
                artifact_id=r["artifact_id"],
                attempt=r["attempt"],
                prompt_id=r["prompt_id"],
                input_json=r["input_json"],
                output_json=r["output_json"],
                verdict=r["verdict"],
                verdict_issues_json=r["verdict_issues_json"],
                model_provider=r["model_provider"],
                model_id=r["model_id"],
                tokens_in=r["tokens_in"],
                tokens_out=r["tokens_out"],
                cost_usd_estimated=r["cost_usd_estimated"],
                latency_ms=r["latency_ms"],
                parent_trace_id=r["parent_trace_id"],
                status=r["status"],
                error_class=r["error_class"],
                error_message=r["error_message"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]
