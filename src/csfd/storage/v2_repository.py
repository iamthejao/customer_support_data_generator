"""Data-access layer for the deterministic-pipeline tables.

Four explicit datasets per run:

* ``problems``           — Problem Database rows (Phase 1 output)
* ``incoming_requests``  — denormalized customer requests (Phase 2)
* ``resolutions``        — multi-turn agent/customer transcripts, one per request
* ``lineage``            — explicit problem → request → resolution traceability

All ``create()`` methods on these repos use ``INSERT OR IGNORE`` keyed by a
stable UID, so a LangGraph checkpointer resume that re-executes a completed
node is a no-op rather than a UNIQUE-constraint error. The id returned on
collision is read back from the existing row so callers always get a valid
foreign key.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from csfd.storage.db import Database


@dataclass(slots=True)
class ProblemV2Record:
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


class ProblemV2Repo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, p: ProblemV2Record) -> None:
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

    def list_for_run(self, run_id: str) -> list[ProblemV2Record]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM problems WHERE run_id = ? ORDER BY created_at, id",
                (run_id,),
            ).fetchall()
        return [_row_to_problem_v2(r) for r in rows]


def _row_to_problem_v2(r: sqlite3.Row) -> ProblemV2Record:
    hints = json.loads(r["resolution_hints_json"]) if r["resolution_hints_json"] else {}
    return ProblemV2Record(
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

    def count_for_run(self, run_id: str) -> int:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM lineage WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["n"])
