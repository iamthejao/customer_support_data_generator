import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from csfd.agents.tracing import record_agent_trace
from csfd.storage.db import Database
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
