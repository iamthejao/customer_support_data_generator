import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import jinja2
import pytest
from pydantic import BaseModel

from csfd.agents.base import AgentContext, AgentRole
from csfd.agents.tracing import ParentLink, TracingAdapter, record_agent_trace
from csfd.prompts.registry import PromptHandle
from csfd.storage.db import Database
from csfd.storage.db_async import AsyncDatabase
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import AgentTraceRepo, RunRecord, RunRepo


def test_record_agent_trace_writes_row(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(
        RunRecord(
            id=run_id,
            phase="phase1",
            parent_run_id=None,
            status="running",
            started_at=datetime.now(UTC),
            completed_at=None,
            run_seed=1,
            pipeline_version="0.1.0",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    trace_id = record_agent_trace(
        repo=AgentTraceRepo(db),
        run_id=run_id,
        thread_id=run_id,
        node_name="problem_brainstorm",
        agent_role="generator",
        artifact_type="problem",
        artifact_id=None,
        attempt=0,
        prompt_id="abc123abc123",
        input_obj={"x": 1},
        output_obj={"y": 2},
        verdict=None,
        verdict_issues=None,
        model_provider="fake",
        model_id="fake-v1",
        tokens_in=10,
        tokens_out=20,
        cost_usd=0.001,
        latency_ms=42,
        parent_trace_id=None,
        status="ok",
        error_class=None,
        error_message=None,
    )
    assert trace_id
    rows = AgentTraceRepo(db).list_for_run(run_id)
    assert len(rows) == 1
    assert rows[0].agent_role == "generator"
    assert json.loads(rows[0].input_json)["x"] == 1


class _Out(BaseModel):
    x: int


class _StubInner(AgentRole):
    """Minimal inner agent that sets last_tokens_* before returning."""

    def __init__(self, tokens_in: int | None, tokens_out: int | None) -> None:
        self.name = "stub"
        env = jinja2.Environment(autoescape=False)
        self.prompt = PromptHandle(
            name="stub",
            template=env.from_string("stub"),
            source="stub",
            version="stubver",
        )
        self.provider = "fake"
        self.model_id = "stub-model"
        self.last_tokens_in = tokens_in
        self.last_tokens_out = tokens_out

    async def invoke(self, ctx: AgentContext) -> BaseModel:
        return _Out(x=1)


@pytest.mark.asyncio
async def test_tracing_adapter_records_token_counts(tmp_db_path: Path) -> None:
    """TracingAdapter must propagate the inner agent's last_tokens_* into the trace row."""
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
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
    adapter = TracingAdapter(
        inner=_StubInner(tokens_in=42, tokens_out=11),
        db=db,
        adb=AsyncDatabase(tmp_db_path),
        run_id=run_id,
        node_name="problem_brainstorm",
        artifact_type="problem",
        artifact_id="p0",
        role="generator",
        parent_link=ParentLink(),
    )
    await adapter.invoke(AgentContext(inputs={}))
    rows = AgentTraceRepo(db).list_for_run(run_id)
    assert len(rows) == 1
    assert rows[0].tokens_in == 42
    assert rows[0].tokens_out == 11


@pytest.mark.asyncio
async def test_tracing_adapter_token_counts_default_none(tmp_db_path: Path) -> None:
    """If the inner agent has no usage metadata, the trace row stores NULL tokens."""
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
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
    adapter = TracingAdapter(
        inner=_StubInner(tokens_in=None, tokens_out=None),
        db=db,
        adb=AsyncDatabase(tmp_db_path),
        run_id=run_id,
        node_name="problem_brainstorm",
        artifact_type="problem",
        artifact_id="p0",
        role="generator",
        parent_link=ParentLink(),
    )
    await adapter.invoke(AgentContext(inputs={}))
    rows = AgentTraceRepo(db).list_for_run(run_id)
    assert rows[0].tokens_in is None
    assert rows[0].tokens_out is None
