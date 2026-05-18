from datetime import UTC, datetime
from pathlib import Path

import pytest

from csfd.storage.db import Database
from csfd.storage.db_async import AsyncDatabase
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    AgentTraceRecord,
    AgentTraceRepo,
    IncomingRequestRecord,
    IncomingRequestRepo,
    LineageRecord,
    LineageRepo,
    ProblemRecord,
    ProblemRepo,
    ResolutionRecord,
    ResolutionRepo,
    RunRecord,
    RunRepo,
)


def _make_run(db: Database, run_id: str = "r1") -> None:
    apply_migrations(db)
    RunRepo(db).create(
        RunRecord(
            id=run_id,
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


@pytest.mark.asyncio
async def test_async_database_round_trip(tmp_path: Path) -> None:
    db = AsyncDatabase(tmp_path / "runs.sqlite")
    async with db.connect() as conn:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO t (name) VALUES (?)", ("alice",))
    async with db.connect() as conn:
        cur = await conn.execute("SELECT name FROM t")
        rows = await cur.fetchall()
    assert [r["name"] for r in rows] == ["alice"]


@pytest.mark.asyncio
async def test_async_database_rolls_back_on_exception(tmp_path: Path) -> None:
    db = AsyncDatabase(tmp_path / "runs.sqlite")
    async with db.connect() as conn:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
    with pytest.raises(RuntimeError):
        async with db.connect() as conn:
            await conn.execute("INSERT INTO t (name) VALUES ('bob')")
            raise RuntimeError("boom")
    async with db.connect() as conn:
        cur = await conn.execute("SELECT COUNT(*) AS n FROM t")
        row = await cur.fetchone()
    assert row is not None
    assert row["n"] == 0


@pytest.mark.asyncio
async def test_run_repo_acreate_and_aupdate(tmp_path: Path) -> None:
    db = Database(tmp_path / "runs.sqlite")
    apply_migrations(db)
    adb = AsyncDatabase(tmp_path / "runs.sqlite")
    await RunRepo(db).acreate(
        adb,
        RunRecord(
            id="r2",
            phase="full",
            parent_run_id=None,
            status="running",
            started_at=datetime.now(UTC),
            completed_at=None,
            run_seed=1,
            pipeline_version="t",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        ),
    )
    await RunRepo(db).aupdate_status(adb, "r2", status="completed", stats={"x": 1})
    rec = RunRepo(db).get("r2")
    assert rec.status == "completed"
    assert rec.stats_json is not None


@pytest.mark.asyncio
async def test_agent_trace_repo_acreate(tmp_path: Path) -> None:
    db = Database(tmp_path / "runs.sqlite")
    _make_run(db)
    adb = AsyncDatabase(tmp_path / "runs.sqlite")
    await AgentTraceRepo(db).acreate(
        adb,
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
            model_provider="fake",
            model_id="stub",
            tokens_in=10,
            tokens_out=5,
            cost_usd_estimated=None,
            latency_ms=12,
            parent_trace_id=None,
            status="ok",
            error_class=None,
            error_message=None,
            created_at=datetime.now(UTC),
        ),
    )
    rows = AgentTraceRepo(db).list_for_run("r1")
    assert len(rows) == 1
    assert rows[0].tokens_in == 10


@pytest.mark.asyncio
async def test_problem_repo_acreate(tmp_path: Path) -> None:
    db = Database(tmp_path / "runs.sqlite")
    _make_run(db)
    adb = AsyncDatabase(tmp_path / "runs.sqlite")
    await ProblemRepo(db).acreate(
        adb,
        ProblemRecord(
            id="r1:p:0000",
            run_id="r1",
            title="t",
            summary="s",
            background="b",
            category="auth",
            complexity="simple",
            resolution_hints={"l1": "x"},
            quality_flag=None,
            created_at=datetime.now(UTC),
        ),
    )
    assert len(ProblemRepo(db).list_for_run("r1")) == 1


@pytest.mark.asyncio
async def test_incoming_request_repo_acreate(tmp_path: Path) -> None:
    db = Database(tmp_path / "runs.sqlite")
    _make_run(db)
    ProblemRepo(db).create(
        ProblemRecord(
            id="r1:p:0000",
            run_id="r1",
            title="t",
            summary="s",
            background="b",
            category="auth",
            complexity="simple",
            resolution_hints={"l1": "x"},
            quality_flag=None,
            created_at=datetime.now(UTC),
        )
    )
    adb = AsyncDatabase(tmp_path / "runs.sqlite")
    ir_id = await IncomingRequestRepo(db).acreate(
        adb,
        IncomingRequestRecord(
            request_uid="r1:000000:req",
            run_id="r1",
            problem_id="r1:p:0000",
            ticket_type="l1",
            customer_name="c",
            customer_tier="standard",
            customer_tone="neutral",
            channel="email",
            subject="s",
            body="b",
            quality_flag=None,
            created_at=datetime.now(UTC),
        ),
    )
    assert ir_id > 0
    # Idempotent: second call returns the same id, doesn't create duplicate.
    ir_id_2 = await IncomingRequestRepo(db).acreate(
        adb,
        IncomingRequestRecord(
            request_uid="r1:000000:req",
            run_id="r1",
            problem_id="r1:p:0000",
            ticket_type="l1",
            customer_name="c",
            customer_tier="standard",
            customer_tone="neutral",
            channel="email",
            subject="s",
            body="b",
            quality_flag=None,
            created_at=datetime.now(UTC),
        ),
    )
    assert ir_id_2 == ir_id
    assert IncomingRequestRepo(db).count_for_run("r1") == 1


@pytest.mark.asyncio
async def test_resolution_repo_acreate(tmp_path: Path) -> None:
    db = Database(tmp_path / "runs.sqlite")
    _make_run(db)
    ProblemRepo(db).create(
        ProblemRecord(
            id="r1:p:0000",
            run_id="r1",
            title="t",
            summary="s",
            background="b",
            category="auth",
            complexity="simple",
            resolution_hints={"l1": "x"},
            quality_flag=None,
            created_at=datetime.now(UTC),
        )
    )
    ir_id = IncomingRequestRepo(db).create(
        IncomingRequestRecord(
            request_uid="r1:000000:req",
            run_id="r1",
            problem_id="r1:p:0000",
            ticket_type="l1",
            customer_name="c",
            customer_tier="standard",
            customer_tone="neutral",
            channel="email",
            subject="s",
            body="b",
            quality_flag=None,
            created_at=datetime.now(UTC),
        )
    )
    adb = AsyncDatabase(tmp_path / "runs.sqlite")
    res_id = await ResolutionRepo(db).acreate(
        adb,
        ResolutionRecord(
            resolution_uid="r1:000000:res",
            run_id="r1",
            incoming_request_id=ir_id,
            problem_id="r1:p:0000",
            ticket_type="l1",
            turns=[{"speaker": "customer", "name": "c", "content": "hi"}],
            turn_count=1,
            resolved=True,
            quality_flag=None,
            created_at=datetime.now(UTC),
        ),
    )
    assert res_id > 0
    assert ResolutionRepo(db).count_for_run("r1") == 1


@pytest.mark.asyncio
async def test_lineage_repo_acreate_and_aupdate(tmp_path: Path) -> None:
    db = Database(tmp_path / "runs.sqlite")
    _make_run(db)
    ProblemRepo(db).create(
        ProblemRecord(
            id="r1:p:0000",
            run_id="r1",
            title="t",
            summary="s",
            background="b",
            category="auth",
            complexity="simple",
            resolution_hints={"l1": "x"},
            quality_flag=None,
            created_at=datetime.now(UTC),
        )
    )
    adb = AsyncDatabase(tmp_path / "runs.sqlite")
    await LineageRepo(db).acreate(
        adb,
        LineageRecord(
            ticket_uid="r1:000000",
            run_id="r1",
            slot_index=0,
            problem_id="r1:p:0000",
            ticket_type="l1",
            customer_tier="standard",
            customer_tone="neutral",
            incoming_request_id=None,
            resolution_id=None,
            created_at=datetime.now(UTC),
        ),
    )
    assert LineageRepo(db).count_for_run("r1") == 1
    ir_id = IncomingRequestRepo(db).create(
        IncomingRequestRecord(
            request_uid="r1:000000:req",
            run_id="r1",
            problem_id="r1:p:0000",
            ticket_type="l1",
            customer_name="c",
            customer_tier="standard",
            customer_tone="neutral",
            channel="email",
            subject="s",
            body="b",
            quality_flag=None,
            created_at=datetime.now(UTC),
        )
    )
    await LineageRepo(db).aupdate_links(adb, "r1:000000", incoming_request_id=ir_id)
    with db.connect() as conn:
        row = conn.execute(
            "SELECT incoming_request_id FROM lineage WHERE ticket_uid='r1:000000'"
        ).fetchone()
    assert row["incoming_request_id"] == ir_id
