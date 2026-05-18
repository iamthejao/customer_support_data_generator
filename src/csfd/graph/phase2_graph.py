"""Phase 2 subgraph — plan-driven resolution generation.

This module defines the LangGraph subgraph that ports
:func:`csfd.pipeline.generate_resolutions` onto LangGraph. The subgraph first
builds a deterministic allocation plan (via :func:`csfd.allocator.build_allocation_plan`)
and pre-records the lineage rows; then, for each slot, flows through a
generator -> (optional) checker -> commit cycle with bounded retries on
checker failure. The subgraph shares
:class:`csfd.graph.pipeline_graph.PipelineState` with the parent graph and the
Phase 1 subgraph; no input/output mapping is needed.

Subgraph topology:

    START -> build_allocation_plan -> generate_resolution
                                          |
                                          v
                  _route_after_generate_resolution
                    /                            \\
        validate_resolution            _mark_validation_skipped
                |                                  |
        _route_after_validate_resolution           |
        /          |           \\                  |
      pass       retry       exhausted             |
        |          |             |                 |
        |    _bump_retry    _mark_exhausted        |
        |          |             |                 |
        |          v             v                 v
        |   generate_resolution  commit_resolution <-+
        |                              ^
        +------------------------------+
                                       |
                       _route_after_commit_resolution
                              /                  \\
                            next                 done
                              |                    \\
                       generate_resolution          END
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from csfd.agents.base import AgentContext, Verdict
from csfd.agents.factory import AgentFactory
from csfd.agents.tracing import ParentLink, TracingAdapter
from csfd.allocator import ProblemRef, build_allocation_plan
from csfd.graph.pipeline_graph import PipelineState, _PlanSlot
from csfd.pipeline import ResolutionOutput
from csfd.settings import AppSettings
from csfd.storage.db import Database
from csfd.storage.v2_repository import (
    IncomingRequestRecord,
    IncomingRequestRepo,
    LineageRecord,
    LineageRepo,
    ResolutionRecord,
    ResolutionRepo,
)
from csfd.ticket_types.definitions import ProblemComplexity

# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


async def build_allocation_plan_node(
    state: PipelineState,
    *,
    db: Database,
    settings: AppSettings,
) -> dict[str, Any]:
    """Build the deterministic allocation plan and pre-record lineage rows."""
    tickets_cfg = settings.tickets
    plan = build_allocation_plan(
        total=tickets_cfg.total,
        type_proportions=tickets_cfg.type_proportions,
        tier_proportions=tickets_cfg.tier_proportions,
        tone_proportions_per_type=tickets_cfg.tone_proportions_per_type,
        problems=[
            ProblemRef(id=p.id, complexity=ProblemComplexity(p.complexity))
            for p in state.problems_committed
        ],
        assignment_strategy=tickets_cfg.assignment_strategy,
        seed=settings.pipeline.run_seed or 0,
    )

    lineage_repo = LineageRepo(db)
    for slot in plan.slots:
        lineage_repo.create(
            LineageRecord(
                ticket_uid=f"{state.run_id}:{slot.index:06d}",
                run_id=state.run_id,
                slot_index=slot.index,
                problem_id=slot.problem_id,
                ticket_type=slot.ticket_type.value,
                customer_tier=slot.tier,
                customer_tone=slot.tone,
                incoming_request_id=None,
                resolution_id=None,
                created_at=datetime.now(UTC),
            )
        )

    pydantic_slots = [
        _PlanSlot(
            index=s.index,
            problem_id=s.problem_id,
            ticket_type=s.ticket_type,
            tier=s.tier,
            tone=s.tone,
        )
        for s in plan.slots
    ]
    return {"plan_slots": pydantic_slots, "slot_index": 0}


async def generate_resolution_node(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
    settings: AppSettings,
) -> dict[str, Any]:
    """Invoke the resolution generator for the current slot."""
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"

    problem_by_id = {p.id: p for p in state.problems_committed}
    problem = problem_by_id[slot.problem_id]
    customer_name = f"Customer-{slot.tier}-{slot.index:04d}"
    agent_name = f"Agent-{slot.ticket_type.value}-{slot.index:04d}"
    target_turns = settings.tickets.turns_per_type[slot.ticket_type.value]

    inputs: dict[str, Any] = {
        "company_name": state.company.name,
        "ticket_type": slot.ticket_type.value,
        "customer_tier": slot.tier,
        "customer_tone": slot.tone,
        "customer_name": customer_name,
        "agent_name": agent_name,
        "target_turn_count": target_turns,
        "problem": {
            "id": problem.id,
            "title": problem.title,
            "summary": problem.summary,
            "background": problem.background,
            "category": problem.category,
            "complexity": problem.complexity,
            "resolution_hint": problem.resolution_hints.get(slot.ticket_type.value, ""),
        },
    }

    link = ParentLink()
    generator = factory.build_generator(
        name="resolution_generator",
        prompt_name="phase2.resolution_generator",
        output_schema_factory=lambda: ResolutionOutput,
    )
    traced_gen = TracingAdapter(
        inner=generator,
        db=db,
        run_id=state.run_id,
        node_name="resolution_generator",
        artifact_type="resolution",
        artifact_id=ticket_uid,
        role="generator",
        parent_link=link,
    )
    # Feed the prior checker verdict back so the prompt's "prior attempt
    # failed validation" block has actual issues to point at on retry.
    prior_verdicts = [state.last_verdict] if state.last_verdict is not None else []
    result = await traced_gen.invoke(
        AgentContext(
            inputs=inputs, retry_attempt=state.retry_attempt, prior_verdicts=prior_verdicts
        )
    )
    assert isinstance(result, ResolutionOutput)
    return {
        "current_resolution_draft": result,
        "last_generator_trace_id": link.last_generator_trace_id,
    }


async def validate_resolution_node(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
    settings: AppSettings,
) -> dict[str, Any]:
    """Run the combined resolution checker against the current draft."""
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    assert state.current_resolution_draft is not None

    problem_by_id = {p.id: p for p in state.problems_committed}
    problem = problem_by_id[slot.problem_id]
    customer_name = f"Customer-{slot.tier}-{slot.index:04d}"
    agent_name = f"Agent-{slot.ticket_type.value}-{slot.index:04d}"
    target_turns = settings.tickets.turns_per_type[slot.ticket_type.value]

    inputs: dict[str, Any] = {
        "company_name": state.company.name,
        "ticket_type": slot.ticket_type.value,
        "customer_tier": slot.tier,
        "customer_tone": slot.tone,
        "customer_name": customer_name,
        "agent_name": agent_name,
        "target_turn_count": target_turns,
        "problem": {
            "id": problem.id,
            "title": problem.title,
            "summary": problem.summary,
            "background": problem.background,
            "category": problem.category,
            "complexity": problem.complexity,
            "resolution_hint": problem.resolution_hints.get(slot.ticket_type.value, ""),
        },
        "candidate": state.current_resolution_draft.model_dump(),
    }

    link = ParentLink(last_generator_trace_id=state.last_generator_trace_id)
    checker = factory.build_checker(
        name="combined_resolution_check",
        prompt_name="phase2.resolution_combined_check",
    )
    traced_check = TracingAdapter(
        inner=checker,
        db=db,
        run_id=state.run_id,
        node_name="combined_resolution_check",
        artifact_type="resolution",
        artifact_id=ticket_uid,
        role="checker",
        parent_link=link,
    )
    verdict = await traced_check.invoke(
        AgentContext(inputs=inputs, retry_attempt=state.retry_attempt, prior_verdicts=[])
    )
    assert isinstance(verdict, Verdict)
    return {
        "last_verdict": verdict,
        "last_quality_flag": None if verdict.passed else state.last_quality_flag,
    }


async def commit_resolution_node(
    state: PipelineState,
    *,
    db: Database,
) -> dict[str, Any]:
    """Persist incoming request + resolution and backfill the lineage row."""
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    assert state.current_resolution_draft is not None
    draft = state.current_resolution_draft

    problem_by_id = {p.id: p for p in state.problems_committed}
    problem = problem_by_id[slot.problem_id]
    customer_name = f"Customer-{slot.tier}-{slot.index:04d}"

    ir_id = IncomingRequestRepo(db).create(
        IncomingRequestRecord(
            request_uid=f"{ticket_uid}:req",
            run_id=state.run_id,
            problem_id=problem.id,
            ticket_type=slot.ticket_type.value,
            customer_name=customer_name,
            customer_tier=slot.tier,
            customer_tone=slot.tone,
            channel="email",
            subject=draft.subject,
            body=draft.body,
            quality_flag=state.last_quality_flag,
            created_at=datetime.now(UTC),
        )
    )
    res_id = ResolutionRepo(db).create(
        ResolutionRecord(
            resolution_uid=f"{ticket_uid}:res",
            run_id=state.run_id,
            incoming_request_id=ir_id,
            problem_id=problem.id,
            ticket_type=slot.ticket_type.value,
            turns=[t.model_dump() for t in draft.turns],
            turn_count=len(draft.turns),
            resolved=draft.resolved,
            quality_flag=state.last_quality_flag,
            created_at=datetime.now(UTC),
        )
    )
    LineageRepo(db).update_links(ticket_uid, incoming_request_id=ir_id, resolution_id=res_id)

    return {
        "slot_index": state.slot_index + 1,
        "retry_attempt": 0,
        "current_resolution_draft": None,
        "last_verdict": None,
        "last_quality_flag": None,
        "last_generator_trace_id": None,
    }


# --------------------------------------------------------------------------- #
# Small helper nodes (retry / quality-flag bookkeeping)
# --------------------------------------------------------------------------- #


async def _bump_retry_node(state: PipelineState) -> dict[str, Any]:
    """Increment the per-artifact retry counter before re-entering generation."""
    return {"retry_attempt": state.retry_attempt + 1}


async def _mark_exhausted_node(state: PipelineState) -> dict[str, Any]:
    """Stamp the retries-exhausted warning before committing a failed draft."""
    return {"last_quality_flag": "warning:retries_exhausted"}


async def _mark_validation_skipped_node(state: PipelineState) -> dict[str, Any]:
    """Stamp the validation-skipped warning when checks are disabled."""
    return {"last_quality_flag": "warning:validation_skipped"}


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #


def _route_after_generate_resolution(state: PipelineState) -> Literal["validate", "commit"]:
    return "validate" if state.validation_enabled else "commit"


def _route_after_validate_resolution(
    state: PipelineState,
) -> Literal["pass", "retry", "exhausted"]:
    assert state.last_verdict is not None, "validate_resolution_node must set last_verdict"
    if state.last_verdict.passed:
        return "pass"
    if state.retry_attempt >= state.max_retries:
        return "exhausted"
    return "retry"


def _route_after_commit_resolution(state: PipelineState) -> Literal["next", "done"]:
    return "next" if state.slot_index < len(state.plan_slots) else "done"


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def build_phase2_subgraph(
    *,
    factory: AgentFactory,
    db: Database,
    settings: AppSettings,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the Phase 2 (resolution generation) subgraph.

    Returns a graph that, given a ``PipelineState`` with ``problems_committed``
    populated by Phase 1, builds a deterministic allocation plan and generates
    one resolution per slot, persisting incoming requests, resolutions, and
    backfilling pre-recorded lineage rows. No checkpointer is attached here —
    the parent graph owns checkpoint persistence.
    """
    g: StateGraph[PipelineState, Any, PipelineState, PipelineState] = StateGraph(PipelineState)
    g.add_node(
        "build_allocation_plan",
        partial(build_allocation_plan_node, db=db, settings=settings),
    )
    g.add_node(
        "generate_resolution",
        partial(generate_resolution_node, factory=factory, db=db, settings=settings),
    )
    g.add_node(
        "validate_resolution",
        partial(validate_resolution_node, factory=factory, db=db, settings=settings),
    )
    g.add_node("commit_resolution", partial(commit_resolution_node, db=db))
    g.add_node("_bump_retry", _bump_retry_node)
    g.add_node("_mark_exhausted", _mark_exhausted_node)
    g.add_node("_mark_validation_skipped", _mark_validation_skipped_node)

    g.add_edge(START, "build_allocation_plan")
    g.add_edge("build_allocation_plan", "generate_resolution")
    g.add_conditional_edges(
        "generate_resolution",
        _route_after_generate_resolution,
        {
            "validate": "validate_resolution",
            "commit": "_mark_validation_skipped",
        },
    )
    g.add_edge("_mark_validation_skipped", "commit_resolution")
    g.add_conditional_edges(
        "validate_resolution",
        _route_after_validate_resolution,
        {
            "pass": "commit_resolution",
            "retry": "_bump_retry",
            "exhausted": "_mark_exhausted",
        },
    )
    g.add_edge("_bump_retry", "generate_resolution")
    g.add_edge("_mark_exhausted", "commit_resolution")
    g.add_conditional_edges(
        "commit_resolution",
        _route_after_commit_resolution,
        {
            "next": "generate_resolution",
            "done": END,
        },
    )

    return g.compile()
