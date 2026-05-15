from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    AgentTraceRecord,
    AgentTraceRepo,
    RunRecord,
    RunRepo,
)


def test_create_and_filter_traces(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="running",
        started_at=datetime.now(timezone.utc),  # noqa: UP017
        completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None, config_snapshot_json="{}",
        stats_json=None, error_summary=None,
    ))
    repo = AgentTraceRepo(db)
    for role in ("generator", "consistency", "background", "scenario"):
        repo.create(AgentTraceRecord(
            id=str(uuid4()), run_id=run_id, thread_id=run_id,
            node_name=f"{role}_node", agent_role=role,
            artifact_type="problem", artifact_id=None, attempt=0,
            prompt_id="abc123", input_json="{}", output_json=None,
            verdict=None, verdict_issues_json=None,
            model_provider="fake", model_id="fake-v1",
            tokens_in=10, tokens_out=20, cost_usd_estimated=0.001,
            latency_ms=100, parent_trace_id=None,
            status="ok", error_class=None, error_message=None,
            created_at=datetime.now(timezone.utc),  # noqa: UP017
        ))
    rows = repo.list_for_run(run_id)
    assert len(rows) == 4
    only_gen = repo.list_for_run(run_id, agent_role="generator")
    assert len(only_gen) == 1
