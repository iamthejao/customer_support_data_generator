"""Data-access layer for all SQLite tables backing the pipeline.

Six repos, two table families:

* Run lifecycle / audit:
    - ``RunRepo``         — one ``runs`` row per pipeline invocation.
    - ``AgentTraceRepo``  — one ``agent_traces`` row per LLM call.

* Per-artifact pipeline tables (one row per artifact, written by commit nodes):
    - ``ProblemRepo``           — Phase 1 Problem Database rows.
    - ``IncomingRequestRepo``   — Phase 2 denormalized customer requests.
    - ``ResolutionRepo``        — Phase 2 multi-turn conversations.
    - ``LineageRepo``           — problem → request → resolution traceability.

All ``create()`` methods use ``INSERT OR IGNORE`` keyed by a stable UID, so a
LangGraph checkpointer resume that re-executes a committed node is a no-op
rather than a UNIQUE-constraint error. The id returned on collision is read
back from the existing row so callers always get a valid foreign key.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from csfd.storage.db import Database

if TYPE_CHECKING:
    from csfd.storage.db_async import AsyncDatabase


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

    async def acreate(self, adb: AsyncDatabase, run: RunRecord) -> None:
        async with adb.connect() as conn:
            await conn.execute(
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

    async def aupdate_status(
        self,
        adb: AsyncDatabase,
        run_id: str,
        *,
        status: str,
        stats: dict[str, Any] | None = None,
        error_summary: str | None = None,
        completed: bool = True,
    ) -> None:
        async with adb.connect() as conn:
            await conn.execute(
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

    async def acreate(self, adb: AsyncDatabase, t: AgentTraceRecord) -> None:
        async with adb.connect() as conn:
            await conn.execute(
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


# --------------------------------------------------------------------------- #
# Per-artifact pipeline tables
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ProblemRecord:
    id: str
    run_id: str
    title: str
    summary: str
    background: str
    category: str
    complexity: str
    resolution_hints: dict[str, str]
    quality_flag: str | None
    created_at: datetime


class ProblemRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, p: ProblemRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO problems
                  (id, run_id, title, summary, background, category,
                   complexity, resolution_hints_json, quality_flag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p.id,
                    p.run_id,
                    p.title,
                    p.summary,
                    p.background,
                    p.category,
                    p.complexity,
                    json.dumps(p.resolution_hints, ensure_ascii=False),
                    p.quality_flag,
                    p.created_at.isoformat(),
                ),
            )

    async def acreate(self, adb: AsyncDatabase, p: ProblemRecord) -> None:
        async with adb.connect() as conn:
            await conn.execute(
                """
                INSERT OR IGNORE INTO problems
                  (id, run_id, title, summary, background, category,
                   complexity, resolution_hints_json, quality_flag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p.id,
                    p.run_id,
                    p.title,
                    p.summary,
                    p.background,
                    p.category,
                    p.complexity,
                    json.dumps(p.resolution_hints, ensure_ascii=False),
                    p.quality_flag,
                    p.created_at.isoformat(),
                ),
            )

    def list_for_run(self, run_id: str) -> list[ProblemRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM problems WHERE run_id = ? ORDER BY created_at, id",
                (run_id,),
            ).fetchall()
        return [_row_to_problem(r) for r in rows]


def _row_to_problem(r: sqlite3.Row) -> ProblemRecord:
    hints = json.loads(r["resolution_hints_json"]) if r["resolution_hints_json"] else {}
    return ProblemRecord(
        id=r["id"],
        run_id=r["run_id"],
        title=r["title"],
        summary=r["summary"],
        background=r["background"],
        category=r["category"],
        complexity=r["complexity"],
        resolution_hints=hints,
        quality_flag=r["quality_flag"],
        created_at=datetime.fromisoformat(r["created_at"]),
    )


@dataclass(slots=True)
class IncomingRequestRecord:
    request_uid: str
    run_id: str
    problem_id: str
    ticket_type: str
    customer_name: str
    customer_tier: str
    customer_tone: str
    channel: str
    subject: str
    body: str
    quality_flag: str | None
    created_at: datetime


class IncomingRequestRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, r: IncomingRequestRecord) -> int:
        """Idempotent insert keyed by ``request_uid``. Returns the row id.

        On UNIQUE collision (e.g. checkpointer-driven resume re-executing a
        completed node), the existing row's id is returned so the caller still
        gets a valid FK target.
        """
        with self.db.connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO incoming_requests
                  (request_uid, run_id, problem_id, ticket_type, customer_name,
                   customer_tier, customer_tone, channel, subject, body,
                   quality_flag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.request_uid,
                    r.run_id,
                    r.problem_id,
                    r.ticket_type,
                    r.customer_name,
                    r.customer_tier,
                    r.customer_tone,
                    r.channel,
                    r.subject,
                    r.body,
                    r.quality_flag,
                    r.created_at.isoformat(),
                ),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                "SELECT id FROM incoming_requests WHERE request_uid = ?",
                (r.request_uid,),
            ).fetchone()
            return int(row["id"])

    async def acreate(self, adb: AsyncDatabase, r: IncomingRequestRecord) -> int:
        """Async sibling of :meth:`create`. Returns the inserted-or-existing row id."""
        async with adb.connect() as conn:
            cur = await conn.execute(
                """
                INSERT OR IGNORE INTO incoming_requests
                  (request_uid, run_id, problem_id, ticket_type, customer_name,
                   customer_tier, customer_tone, channel, subject, body,
                   quality_flag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.request_uid,
                    r.run_id,
                    r.problem_id,
                    r.ticket_type,
                    r.customer_name,
                    r.customer_tier,
                    r.customer_tone,
                    r.channel,
                    r.subject,
                    r.body,
                    r.quality_flag,
                    r.created_at.isoformat(),
                ),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            cur2 = await conn.execute(
                "SELECT id FROM incoming_requests WHERE request_uid = ?",
                (r.request_uid,),
            )
            row = await cur2.fetchone()
            assert row is not None
            return int(row["id"])

    def count_for_run(self, run_id: str) -> int:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM incoming_requests WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["n"])

    def list_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM incoming_requests WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]


@dataclass(slots=True)
class ResolutionRecord:
    resolution_uid: str
    run_id: str
    incoming_request_id: int
    problem_id: str
    ticket_type: str
    turns: list[dict[str, Any]]
    turn_count: int
    resolved: bool
    quality_flag: str | None
    created_at: datetime


class ResolutionRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, r: ResolutionRecord) -> int:
        """Idempotent insert keyed by ``resolution_uid``. Returns the row id.

        On UNIQUE collision (e.g. checkpointer-driven resume), the existing
        row's id is returned.
        """
        with self.db.connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO resolutions
                  (resolution_uid, run_id, incoming_request_id, problem_id,
                   ticket_type, turns_json, turn_count, resolved,
                   quality_flag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.resolution_uid,
                    r.run_id,
                    r.incoming_request_id,
                    r.problem_id,
                    r.ticket_type,
                    json.dumps(r.turns, ensure_ascii=False),
                    r.turn_count,
                    1 if r.resolved else 0,
                    r.quality_flag,
                    r.created_at.isoformat(),
                ),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                "SELECT id FROM resolutions WHERE resolution_uid = ?",
                (r.resolution_uid,),
            ).fetchone()
            return int(row["id"])

    async def acreate(self, adb: AsyncDatabase, r: ResolutionRecord) -> int:
        """Async sibling of :meth:`create`. Returns the inserted-or-existing row id."""
        async with adb.connect() as conn:
            cur = await conn.execute(
                """
                INSERT OR IGNORE INTO resolutions
                  (resolution_uid, run_id, incoming_request_id, problem_id,
                   ticket_type, turns_json, turn_count, resolved,
                   quality_flag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.resolution_uid,
                    r.run_id,
                    r.incoming_request_id,
                    r.problem_id,
                    r.ticket_type,
                    json.dumps(r.turns, ensure_ascii=False),
                    r.turn_count,
                    1 if r.resolved else 0,
                    r.quality_flag,
                    r.created_at.isoformat(),
                ),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            cur2 = await conn.execute(
                "SELECT id FROM resolutions WHERE resolution_uid = ?",
                (r.resolution_uid,),
            )
            row = await cur2.fetchone()
            assert row is not None
            return int(row["id"])

    def count_for_run(self, run_id: str) -> int:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM resolutions WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["n"])


@dataclass(slots=True)
class LineageRecord:
    ticket_uid: str
    run_id: str
    slot_index: int
    problem_id: str
    ticket_type: str
    customer_tier: str
    customer_tone: str
    incoming_request_id: int | None
    resolution_id: int | None
    created_at: datetime


class LineageRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, r: LineageRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO lineage
                  (ticket_uid, run_id, slot_index, problem_id, ticket_type,
                   customer_tier, customer_tone, incoming_request_id,
                   resolution_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.ticket_uid,
                    r.run_id,
                    r.slot_index,
                    r.problem_id,
                    r.ticket_type,
                    r.customer_tier,
                    r.customer_tone,
                    r.incoming_request_id,
                    r.resolution_id,
                    r.created_at.isoformat(),
                ),
            )

    def update_links(
        self,
        ticket_uid: str,
        *,
        incoming_request_id: int | None = None,
        resolution_id: int | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if incoming_request_id is not None:
            sets.append("incoming_request_id = ?")
            params.append(incoming_request_id)
        if resolution_id is not None:
            sets.append("resolution_id = ?")
            params.append(resolution_id)
        if not sets:
            return
        params.append(ticket_uid)
        with self.db.connect() as conn:
            conn.execute(
                f"UPDATE lineage SET {', '.join(sets)} WHERE ticket_uid = ?",
                tuple(params),
            )

    async def acreate(self, adb: AsyncDatabase, r: LineageRecord) -> None:
        async with adb.connect() as conn:
            await conn.execute(
                """
                INSERT OR IGNORE INTO lineage
                  (ticket_uid, run_id, slot_index, problem_id, ticket_type,
                   customer_tier, customer_tone, incoming_request_id,
                   resolution_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.ticket_uid,
                    r.run_id,
                    r.slot_index,
                    r.problem_id,
                    r.ticket_type,
                    r.customer_tier,
                    r.customer_tone,
                    r.incoming_request_id,
                    r.resolution_id,
                    r.created_at.isoformat(),
                ),
            )

    async def aupdate_links(
        self,
        adb: AsyncDatabase,
        ticket_uid: str,
        *,
        incoming_request_id: int | None = None,
        resolution_id: int | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if incoming_request_id is not None:
            sets.append("incoming_request_id = ?")
            params.append(incoming_request_id)
        if resolution_id is not None:
            sets.append("resolution_id = ?")
            params.append(resolution_id)
        if not sets:
            return
        params.append(ticket_uid)
        async with adb.connect() as conn:
            await conn.execute(
                f"UPDATE lineage SET {', '.join(sets)} WHERE ticket_uid = ?",
                tuple(params),
            )

    def count_for_run(self, run_id: str) -> int:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM lineage WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["n"])
