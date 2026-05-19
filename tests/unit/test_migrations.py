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
    assert set(applied_versions(db)) == {"001", "002", "003", "004", "005"}


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
