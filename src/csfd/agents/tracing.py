"""Helpers to persist agent invocations as ``AgentTraceRecord`` rows.

This module exposes two surfaces:

* :func:`record_agent_trace` — low-level row-writer used by both the
  :class:`TracingAdapter` below and any future caller that wants to emit a
  trace outside the standard adapter flow.
* :class:`TracingAdapter` — wraps an inner :class:`csfd.agents.base.AgentRole`
  (``Generator`` or ``Checker``) and writes one trace row per ``invoke``.
  The adapter is the only hook the deterministic pipeline uses to attach
  persistence to agent calls, keeping ``_run_with_retries`` storage-agnostic.

Generator → checker linkage: a generator adapter and a checker adapter that
target the same artifact share a small ``ParentLink`` so the checker's trace
row points at the generator's trace via ``parent_trace_id``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from csfd.agents.base import AgentContext, AgentRole, Verdict
from csfd.storage.db import Database
from csfd.storage.db_async import AsyncDatabase
from csfd.storage.repository import AgentTraceRecord, AgentTraceRepo


def _build_trace_record(
    *,
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
) -> AgentTraceRecord:
    # Deterministic id so INSERT OR IGNORE is actually idempotent on
    # checkpointer resume. Uniqueness is guaranteed by the
    # (run_id, node_name, artifact_type, artifact_id, agent_role, attempt)
    # tuple — run_id is uuid4 per run, and the rest are unique within a run by
    # construction.
    key = f"{run_id}|{node_name}|{artifact_type}|{artifact_id}|{agent_role}|{attempt}"
    trace_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return AgentTraceRecord(
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
    """Sync writer kept for non-graph callers (tests, future tools)."""
    record = _build_trace_record(
        run_id=run_id,
        thread_id=thread_id,
        node_name=node_name,
        agent_role=agent_role,
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        attempt=attempt,
        prompt_id=prompt_id,
        input_obj=input_obj,
        output_obj=output_obj,
        verdict=verdict,
        verdict_issues=verdict_issues,
        model_provider=model_provider,
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        parent_trace_id=parent_trace_id,
        status=status,
        error_class=error_class,
        error_message=error_message,
    )
    repo.create(record)
    return record.id


async def arecord_agent_trace(
    *,
    repo: AgentTraceRepo,
    adb: AsyncDatabase,
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
    """Async writer used by ``TracingAdapter`` inside LangGraph node bodies."""
    record = _build_trace_record(
        run_id=run_id,
        thread_id=thread_id,
        node_name=node_name,
        agent_role=agent_role,
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        attempt=attempt,
        prompt_id=prompt_id,
        input_obj=input_obj,
        output_obj=output_obj,
        verdict=verdict,
        verdict_issues=verdict_issues,
        model_provider=model_provider,
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        parent_trace_id=parent_trace_id,
        status=status,
        error_class=error_class,
        error_message=error_message,
    )
    await repo.acreate(adb, record)
    return record.id


@dataclass
class ParentLink:
    """Shared between paired generator+checker adapters for one artifact.

    The generator adapter writes its trace_id into ``last_generator_trace_id``;
    the checker adapter reads it to populate ``parent_trace_id`` on the next
    checker trace. Reset between artifacts by construction (one new
    ``ParentLink`` per artifact in the pipeline).
    """

    last_generator_trace_id: str | None = field(default=None)


class TracingAdapter(AgentRole):
    """Wrap an inner Generator/Checker so every ``invoke`` writes one trace row.

    The inner agent must expose ``.prompt.version`` (used as ``prompt_id``),
    ``.provider``, and ``.model_id``. Both ``Generator`` and ``Checker``
    provide these attributes.
    """

    def __init__(
        self,
        *,
        inner: AgentRole,
        db: Database,
        adb: AsyncDatabase,
        run_id: str,
        node_name: str,
        artifact_type: str,
        artifact_id: str,
        role: Literal["generator", "checker"],
        parent_link: ParentLink,
    ) -> None:
        self.name = inner.name
        self._inner = inner
        self._repo = AgentTraceRepo(db)
        self._adb = adb
        self._run_id = run_id
        self._node_name = node_name
        self._artifact_type = artifact_type
        self._artifact_id = artifact_id
        self._role = role
        self._parent_link = parent_link

    async def invoke(self, ctx: AgentContext) -> BaseModel:
        parent_trace_id = (
            self._parent_link.last_generator_trace_id if self._role == "checker" else None
        )
        t0 = time.perf_counter()
        result: BaseModel | None = None
        status = "ok"
        error_class: str | None = None
        error_message: str | None = None
        try:
            result = await self._inner.invoke(ctx)
        except ValidationError as e:
            status = "schema_error"
            error_class = type(e).__name__
            error_message = str(e)[:1000]
            raise
        except Exception as e:
            status = "transport_error"
            error_class = type(e).__name__
            error_message = str(e)[:1000]
            raise
        finally:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            verdict_str: str | None = None
            verdict_issues: list[dict[str, Any]] | None = None
            if self._role == "checker" and isinstance(result, Verdict):
                verdict_str = "pass" if result.passed else "fail"
                verdict_issues = [i.model_dump() for i in result.issues]
            inner_any: Any = self._inner
            output_obj = result.model_dump() if isinstance(result, BaseModel) else None
            trace_id = await arecord_agent_trace(
                repo=self._repo,
                adb=self._adb,
                run_id=self._run_id,
                thread_id=f"{self._run_id}:{self._artifact_type}:{self._artifact_id}",
                node_name=self._node_name,
                agent_role=self._role,
                artifact_type=self._artifact_type,
                artifact_id=self._artifact_id,
                attempt=ctx.retry_attempt,
                prompt_id=inner_any.prompt.version,
                input_obj=ctx.inputs,
                output_obj=output_obj,
                verdict=verdict_str,
                verdict_issues=verdict_issues,
                model_provider=inner_any.provider,
                model_id=inner_any.model_id,
                tokens_in=getattr(self._inner, "last_tokens_in", None),
                tokens_out=getattr(self._inner, "last_tokens_out", None),
                cost_usd=None,  # tokens flow now; cost requires a price table — out of scope.
                latency_ms=latency_ms,
                parent_trace_id=parent_trace_id,
                status=status,
                error_class=error_class,
                error_message=error_message,
            )
            if self._role == "generator":
                self._parent_link.last_generator_trace_id = trace_id
        assert result is not None
        return result
