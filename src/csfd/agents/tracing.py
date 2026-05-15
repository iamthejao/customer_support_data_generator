"""Helper to persist a single agent invocation as an AgentTraceRecord."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from csfd.storage.repository import AgentTraceRecord, AgentTraceRepo


def record_agent_trace(
    *,
    repo: AgentTraceRepo,
    run_id: str,
    thread_id: str,
    node_name: str,
    agent_role: str,
    artifact_type: str,
    artifact_id: str | None,
    attempt: int,
    prompt_id: str,
    input_obj: Any,
    output_obj: Any,
    verdict: str | None,
    verdict_issues: Any,
    model_provider: str,
    model_id: str,
    tokens_in: int | None,
    tokens_out: int | None,
    cost_usd: float | None,
    latency_ms: int | None,
    parent_trace_id: str | None,
    status: str,
    error_class: str | None,
    error_message: str | None,
) -> str:
    trace_id = str(uuid4())
    repo.create(
        AgentTraceRecord(
            id=trace_id,
            run_id=run_id,
            thread_id=thread_id,
            node_name=node_name,
            agent_role=agent_role,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            attempt=attempt,
            prompt_id=prompt_id,
            input_json=json.dumps(input_obj, default=str),
            output_json=json.dumps(output_obj, default=str) if output_obj is not None else None,
            verdict=verdict,
            verdict_issues_json=json.dumps(verdict_issues, default=str)
            if verdict_issues is not None
            else None,
            model_provider=model_provider,
            model_id=model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd_estimated=cost_usd,
            latency_ms=latency_ms,
            parent_trace_id=parent_trace_id,
            status=status,
            error_class=error_class,
            error_message=error_message,
            created_at=datetime.now(UTC),
        )
    )
    return trace_id
