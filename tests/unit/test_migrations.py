import sqlite3
import struct
from pathlib import Path

import pytest

from csfd.storage.db import Database
from csfd.storage.migrations.runner import applied_versions, apply_migrations


def test_apply_migrations_creates_all_tables(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    with db.connect() as conn:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    for required in {
        "runs",
        "problems",
        "incoming_requests",
        "resolutions",
        "lineage",
        "agent_traces",
        "schema_migrations",
    }:
        assert required in names


def test_apply_migrations_is_idempotent(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    apply_migrations(db)
    assert set(applied_versions(db)) == {"001", "002", "003", "004", "005", "006", "007", "008"}


def test_agent_traces_accepts_claude_code_cli_provider(tmp_db_path: Path) -> None:
    from datetime import UTC, datetime

    from csfd.storage.repository import AgentTraceRecord, AgentTraceRepo, RunRecord, RunRepo

    db = Database(path=tmp_db_path)
    apply_migrations(db)
    RunRepo(db).create(
        RunRecord(
            id="r1",
            phase="full",
            parent_run_id=None,
            status="running",
            started_at=datetime.now(UTC),
            completed_at=None,
            run_seed=0,
            pipeline_version="t",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    AgentTraceRepo(db).create(
        AgentTraceRecord(
            id="t1",
            run_id="r1",
            thread_id="r1:problem:p0",
            node_name="problem_brainstorm",
            agent_role="generator",
            artifact_type="problem",
            artifact_id="p0",
            attempt=0,
            prompt_id="ph#abc",
            input_json="{}",
            output_json=None,
            verdict=None,
            verdict_issues_json=None,
            model_provider="claude_code_cli",
            model_id="sonnet-4",
            tokens_in=None,
            tokens_out=None,
            cost_usd_estimated=None,
            latency_ms=10,
            parent_trace_id=None,
            status="ok",
            error_class=None,
            error_message=None,
            created_at=datetime.now(UTC),
        )
    )
    with db.connect() as conn:
        row = conn.execute("SELECT model_provider FROM agent_traces WHERE id='t1'").fetchone()
    assert row["model_provider"] == "claude_code_cli"


def test_problems_table_columns(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    with db.connect() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(problems)").fetchall()}
    assert {
        "id",
        "run_id",
        "title",
        "summary",
        "background",
        "category",
        "complexity",
        "resolution_hints_json",
        "quality_flag",
        "created_at",
    }.issubset(cols)


def test_005_problem_embeddings_applies_and_enforces_fk(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='problem_embeddings'"
        ).fetchall()
        assert rows, "problem_embeddings table not created"
        cols = {r[1] for r in conn.execute("PRAGMA table_info(problem_embeddings)").fetchall()}
        assert cols == {"problem_id", "run_id", "model", "dim", "vector", "created_at"}
        idx = {r[1] for r in conn.execute("PRAGMA index_list(problem_embeddings)").fetchall()}
        assert "idx_problem_embeddings_run" in idx

    # FK enforcement: inserting a problem_embeddings row for a missing problem_id must fail.
    blob = struct.pack("<3f", 1.0, 0.0, 0.0)
    with db.connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO problem_embeddings (problem_id, run_id, model, dim, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("missing_problem", "missing_run", "m", 3, blob, "2026-01-01T00:00:00+00:00"),
        )


def test_case_rounds_migration_backfills_existing_rows(
    tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows written before 008 become single-contact cases keyed by their lineage ticket."""
    from csfd.storage.migrations import runner

    all_migrations = runner._discover()
    monkeypatch.setattr(runner, "_discover", lambda: [m for m in all_migrations if m[0] < "008"])
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    with sqlite3.connect(tmp_db_path) as raw:
        raw.executescript(
            """
            INSERT INTO runs (id, phase, status, started_at, run_seed, pipeline_version,
                              config_snapshot_json)
            VALUES ('r', 'full', 'completed', '2026-01-01', 1, 'x', '{}');
            INSERT INTO problems (id, run_id, title, summary, background, category, complexity,
                                  resolution_hints_json, created_at)
            VALUES ('p', 'r', 't', 's', 'b', 'c', 'simple', '{}', '2026-01-01');
            INSERT INTO incoming_requests (id, request_uid, run_id, problem_id, ticket_type,
                                           customer_name, customer_tier, customer_tone,
                                           subject, body, created_at)
            VALUES (1, 'r:000001:req', 'r', 'p', 'l1', 'n', 'standard', 'neutral', 's', 'b',
                    '2026-01-01');
            INSERT INTO resolutions (id, resolution_uid, run_id, incoming_request_id,
                                     problem_id, ticket_type, turns_json, turn_count,
                                     created_at)
            VALUES (1, 'r:000001:res', 'r', 1, 'p', 'l1', '[]', 0, '2026-01-01');
            INSERT INTO lineage (ticket_uid, run_id, slot_index, problem_id, ticket_type,
                                 customer_tier, customer_tone, incoming_request_id,
                                 resolution_id, created_at)
            VALUES ('r:000001', 'r', 1, 'p', 'l1', 'standard', 'neutral', 1, 1, '2026-01-01');
            """
        )
    monkeypatch.setattr(runner, "_discover", lambda: all_migrations)
    assert apply_migrations(db) == ["008"]
    with db.connect() as conn:
        req = conn.execute("SELECT case_uid, round_index FROM incoming_requests").fetchone()
        res = conn.execute(
            "SELECT case_uid, round_index, round_count, channel FROM resolutions"
        ).fetchone()
    assert (req["case_uid"], req["round_index"]) == ("r:000001", 1)
    assert tuple(res) == ("r:000001", 1, 1, "email")
