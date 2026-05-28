"""Phase 2 subgraph — plan-driven turn-based dialogue generation.

This module defines the LangGraph subgraph that generates one customer-service
conversation per allocation slot. It first builds a deterministic allocation
plan (via :func:`csfd.allocator.build_allocation_plan`) and pre-records the
lineage rows; then, for each slot, it runs a turn-by-turn dialogue between two
information-asymmetric agents:

* the **customer** agent sees only customer-observable problem fields
  (symptoms, impact, persona) and the conversation so far;
* the **service** agent sees only root-cause fields (root cause, background,
  summary, resolution hint) and the conversation so far.

Turn 1 is the customer's opening message (the standalone incoming request).
Turns then alternate, starting with the service agent, until whichever speaker
just spoke flags ``done`` — or a hard ``dialogue.turn_cap`` is reached. A
consistency agent then reviews the full transcript and may pass it, pass it
with edits, or fail it (a fail re-rolls the whole conversation within the
existing retry budget). The subgraph shares
:class:`csfd.graph.pipeline_graph.PipelineState` with the parent graph and the
Phase 1 subgraph; no input/output mapping is needed.

Subgraph topology:

    START -> build_allocation_plan -> generate_incoming_request
                                              |
                                              v
                                     generate_agent_turn <------------+
                                              |                       |
                          _route_after_agent_turn                     |
                          /          |           \\                   |
                  generate_customer_turn  cap   consistency           |
                          |          |           |                    |
            _route_after_customer_turn |          |                   |
            /        |        \\       |          |                   |
       agent      cap      consistency |          |                   |
        (back to generate_agent_turn) -+          |                   |
                             |                     |                   |
                        _mark_cap_hit -> _route_consistency_or_skip <--+
                                              |
                          _route_validation_enabled
                            /                    \\
                  validate_conversation     _mark_validation_skipped
                            |                        |
            _route_after_validate_conversation       |
            /         |            \\                |
          pass      retry       exhausted            |
            |         |             |                |
            |   _bump_retry   _mark_exhausted        |
            |  (back to gen)        |                |
            +-----------------> commit_dialogue <-----+
                                    |
                       _route_after_commit_resolution
                              /                  \\
                            next                 done
                              |                    \\
                  generate_incoming_request          END
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from csfd.agents.base import AgentContext, Issue, Verdict
from csfd.agents.factory import AgentFactory
from csfd.agents.tracing import ParentLink, TracingAdapter
from csfd.allocator import ProblemRef, build_allocation_plan
from csfd.graph.pipeline_graph import PipelineState, _PlanSlot
from csfd.pipeline import (
    ConsistencyVerdict,
    DialogueTurnOutput,
    IncomingRequestOutput,
    ResolutionOutput,
    _apply_consistency_edits,
    _assemble_resolution,
)
from csfd.settings import AppSettings
from csfd.storage.db import Database
from csfd.storage.db_async import AsyncDatabase
from csfd.storage.repository import (
    IncomingRequestRecord,
    IncomingRequestRepo,
    LineageRecord,
    LineageRepo,
    ResolutionRecord,
    ResolutionRepo,
)
from csfd.ticket_types.definitions import ProblemComplexity

# --------------------------------------------------------------------------- #
# Input builders (enforce information asymmetry at the prompt boundary)
# --------------------------------------------------------------------------- #


def _render_history(turns: list[DialogueTurnOutput]) -> list[dict[str, str]]:
    """Render accumulated turns into the simple {speaker, content} dicts prompts expect."""
    return [{"speaker": t.speaker, "content": t.content} for t in turns]


def _customer_inputs(state: PipelineState, slot: _PlanSlot) -> dict[str, Any]:
    """Customer-view inputs: symptoms + impact + persona only. No root cause."""
    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    return {
        "company_name": state.company.name,
        "ticket_type": slot.ticket_type.value,
        "customer_name": f"Customer-{slot.tier}-{slot.index:04d}",
        "customer_tier": slot.tier,
        "customer_tone": slot.tone,
        "symptoms": problem.symptoms,
        "customer_impact": problem.customer_impact,
        "category": problem.category,
        "turn_cap": state.dialogue_turn_cap,
    }


def _agent_inputs(state: PipelineState, slot: _PlanSlot) -> dict[str, Any]:
    """Service-view inputs: root cause + diagnostic context. No symptoms list, no tone."""
    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    return {
        "company_name": state.company.name,
        "ticket_type": slot.ticket_type.value,
        "agent_name": f"Agent-{slot.ticket_type.value}-{slot.index:04d}",
        "problem": {
            "title": problem.title,
            "summary": problem.summary,
            "background": problem.background,
            "category": problem.category,
            "fault_domain": problem.fault_domain,
            "root_cause": problem.root_cause,
            "resolution_hint": problem.resolution_hints.get(slot.ticket_type.value, ""),
        },
        "turn_cap": state.dialogue_turn_cap,
    }


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


async def build_allocation_plan_node(
    state: PipelineState,
    *,
    db: Database,
    adb: AsyncDatabase,
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
        await lineage_repo.acreate(
            adb,
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
            ),
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
    return {
        "plan_slots": pydantic_slots,
        "slot_index": 0,
        "dialogue_turn_cap": settings.tickets.dialogue.turn_cap,
    }


async def generate_incoming_request_node(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
    adb: AsyncDatabase,
    settings: AppSettings,
) -> dict[str, Any]:
    """LLM call: customer's opening message. Seeds turn 1 (customer)."""
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    inputs = _customer_inputs(state, slot)

    link = ParentLink()
    generator = factory.build_generator(
        name="incoming_request_generator",
        prompt_name="phase2.incoming_request",
        output_schema_factory=lambda: IncomingRequestOutput,
    )
    traced = TracingAdapter(
        inner=generator,
        db=db,
        adb=adb,
        run_id=state.run_id,
        node_name="incoming_request_generator",
        artifact_type="resolution",
        artifact_id=ticket_uid,
        role="generator",
        parent_link=link,
    )
    result = await traced.invoke(AgentContext(inputs=inputs, retry_attempt=state.retry_attempt))
    assert isinstance(result, IncomingRequestOutput)
    turn0 = DialogueTurnOutput(speaker="customer", content=result.body, done=False)
    return {
        "current_resolution_draft": ResolutionOutput(subject=result.subject, body=result.body),
        "current_dialogue_turns": [turn0],
        "dialogue_last_speaker": "customer",
        "dialogue_done": False,
        "dialogue_end_reason": None,
        "last_generator_trace_id": link.last_generator_trace_id,
    }


async def _generate_turn(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
    adb: AsyncDatabase,
    speaker: Literal["customer", "agent"],
) -> dict[str, Any]:
    """Shared body for the customer / agent turn nodes: one LLM call, one appended turn."""
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    if speaker == "customer":
        node_name, prompt_name = "customer_turn_generator", "phase2.customer_turn"
        inputs = _customer_inputs(state, slot)
    else:
        node_name, prompt_name = "agent_turn_generator", "phase2.agent_turn"
        inputs = _agent_inputs(state, slot)
    inputs["conversation_so_far"] = _render_history(state.current_dialogue_turns)
    inputs["turn_index"] = len(state.current_dialogue_turns) + 1
    # On a re-roll, surface the prior consistency issues to the agent so it can
    # avoid repeating them. Only the agent turn receives them (it drives diagnosis).
    prior_issues = (
        [i.explanation for i in state.last_verdict.issues]
        if (speaker == "agent" and state.last_verdict is not None and not state.last_verdict.passed)
        else []
    )

    link = ParentLink()
    generator = factory.build_generator(
        name=node_name,
        prompt_name=prompt_name,
        output_schema_factory=lambda: DialogueTurnOutput,
    )
    traced = TracingAdapter(
        inner=generator,
        db=db,
        adb=adb,
        run_id=state.run_id,
        node_name=node_name,
        artifact_type="resolution",
        artifact_id=ticket_uid,
        role="generator",
        parent_link=link,
    )
    ctx = AgentContext(
        inputs={**inputs, "prior_issues": prior_issues}, retry_attempt=state.retry_attempt
    )
    result = await traced.invoke(ctx)
    assert isinstance(result, DialogueTurnOutput)
    # Force the speaker to match this node's role; the LLM may fill it but we own it.
    result = result.model_copy(update={"speaker": speaker})
    turns = [*state.current_dialogue_turns, result]
    end_reason = None
    if result.done:
        end_reason = "customer_done" if speaker == "customer" else "agent_done"
    return {
        "current_dialogue_turns": turns,
        "dialogue_last_speaker": speaker,
        "dialogue_done": result.done,
        "dialogue_end_reason": end_reason,
        "last_generator_trace_id": link.last_generator_trace_id,
    }


async def generate_agent_turn_node(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
    adb: AsyncDatabase,
    settings: AppSettings,
) -> dict[str, Any]:
    """LLM call: one service-agent turn."""
    return await _generate_turn(state, factory=factory, db=db, adb=adb, speaker="agent")


async def generate_customer_turn_node(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
    adb: AsyncDatabase,
    settings: AppSettings,
) -> dict[str, Any]:
    """LLM call: one customer turn."""
    return await _generate_turn(state, factory=factory, db=db, adb=adb, speaker="customer")


async def validate_conversation_node(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
    adb: AsyncDatabase,
    settings: AppSettings,
) -> dict[str, Any]:
    """LLM call: consistency review of the full transcript; may return edits."""
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    # _converge_node has already assembled the accumulated turns into the draft.
    assert state.current_resolution_draft is not None
    draft = state.current_resolution_draft

    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    inputs: dict[str, Any] = {
        "company_name": state.company.name,
        "customer_tone": slot.tone,
        "problem": {
            "title": problem.title,
            "complexity": problem.complexity,
            "symptoms": problem.symptoms,
            "root_cause": problem.root_cause,
            "resolution_hint": problem.resolution_hints.get(slot.ticket_type.value, ""),
        },
        "candidate": draft.model_dump(),
    }

    link = ParentLink(last_generator_trace_id=state.last_generator_trace_id)
    checker = factory.build_generator(
        name="conversation_consistency_check",
        prompt_name="phase2.conversation_consistency_check",
        output_schema_factory=lambda: ConsistencyVerdict,
    )
    traced = TracingAdapter(
        inner=checker,
        db=db,
        adb=adb,
        run_id=state.run_id,
        node_name="conversation_consistency_check",
        artifact_type="resolution",
        artifact_id=ticket_uid,
        role="checker",
        parent_link=link,
    )
    verdict = await traced.invoke(AgentContext(inputs=inputs, retry_attempt=state.retry_attempt))
    assert isinstance(verdict, ConsistencyVerdict)

    passed = verdict.status in ("pass", "pass_with_edits")
    edited = verdict.status == "pass_with_edits"
    final_draft = _apply_consistency_edits(draft, verdict) if passed else draft
    quality_flag = state.last_quality_flag
    # An edited transcript is flagged unless a turn-cap-hit already claimed the
    # slot — the cap is the more actionable signal and must not be overwritten.
    if edited and quality_flag != "warning:turn_cap_hit":
        quality_flag = "info:consistency_edited"
    # Surface the consistency issues on a fail so the re-roll's agent turns can
    # see them via AgentContext (see _generate_turn's prior_issues wiring).
    issues = (
        []
        if passed
        else [
            Issue(
                severity="error",
                location="conversation",
                rule_violated="consistency",
                explanation=msg,
            )
            for msg in verdict.issues
        ]
    )
    return {
        "current_resolution_draft": final_draft,
        "current_dialogue_turns": final_draft.turns,
        "last_verdict": Verdict(
            checker="conversation_consistency_check", passed=passed, issues=issues
        ),
        "last_quality_flag": quality_flag,
    }


async def commit_dialogue_node(
    state: PipelineState,
    *,
    db: Database,
    adb: AsyncDatabase,
) -> dict[str, Any]:
    """Persist incoming request + resolution, backfill lineage, reset per-slot state."""
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    assert state.current_resolution_draft is not None
    draft = state.current_resolution_draft

    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    customer_name = f"Customer-{slot.tier}-{slot.index:04d}"

    ir_id = await IncomingRequestRepo(db).acreate(
        adb,
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
        ),
    )
    res_id = await ResolutionRepo(db).acreate(
        adb,
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
        ),
    )
    await LineageRepo(db).aupdate_links(
        adb, ticket_uid, incoming_request_id=ir_id, resolution_id=res_id
    )

    return {
        "slot_index": state.slot_index + 1,
        "retry_attempt": 0,
        "current_resolution_draft": None,
        "current_dialogue_turns": [],
        "dialogue_last_speaker": None,
        "dialogue_done": False,
        "dialogue_end_reason": None,
        "last_verdict": None,
        "last_quality_flag": None,
        "last_generator_trace_id": None,
    }


# --------------------------------------------------------------------------- #
# Small helper nodes (retry / quality-flag / convergence bookkeeping)
# --------------------------------------------------------------------------- #


async def _bump_retry_node(state: PipelineState) -> dict[str, Any]:
    """Increment the per-artifact retry counter before re-rolling the conversation."""
    return {"retry_attempt": state.retry_attempt + 1}


async def _mark_exhausted_node(state: PipelineState) -> dict[str, Any]:
    """Stamp the retries-exhausted warning before committing a failed draft."""
    return {"last_quality_flag": "warning:retries_exhausted"}


async def _mark_validation_skipped_node(state: PipelineState) -> dict[str, Any]:
    """Stamp the validation-skipped warning when checks are disabled.

    A turn-cap-hit flag takes precedence: a capped conversation that also skips
    validation should still surface the cap, the more actionable signal.
    """
    if state.last_quality_flag == "warning:turn_cap_hit":
        return {}
    return {"last_quality_flag": "warning:validation_skipped"}


async def _mark_cap_hit_node(state: PipelineState) -> dict[str, Any]:
    """Stamp the turn-cap warning and pin the end reason when the cap is reached."""
    return {
        "dialogue_end_reason": "cap_hit",
        "last_quality_flag": "warning:turn_cap_hit",
    }


async def _converge_node(state: PipelineState) -> dict[str, Any]:
    """Assemble the finished transcript into the draft before validation routing.

    Both the validate and the skip branches read ``current_resolution_draft`` at
    commit time, so the accumulated turns + derived ``resolved``/``end_reason``
    must be folded in here — not only inside ``validate_conversation_node`` —
    otherwise the validation-disabled path would commit an empty turn list.
    """
    assert state.current_resolution_draft is not None
    draft = _assemble_resolution(
        subject=state.current_resolution_draft.subject,
        body=state.current_resolution_draft.body,
        turns=state.current_dialogue_turns,
        end_reason=state.dialogue_end_reason or "agent_done",
    )
    return {"current_resolution_draft": draft}


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #


def _route_after_build_allocation_plan(state: PipelineState) -> Literal["generate", "done"]:
    """Skip the per-slot loop entirely if no slots were planned (tickets.total == 0)."""
    return "generate" if state.plan_slots else "done"


def _route_after_agent_turn(state: PipelineState) -> Literal["customer", "consistency", "cap"]:
    if state.dialogue_done:
        return "consistency"
    if len(state.current_dialogue_turns) >= state.dialogue_turn_cap:
        return "cap"
    return "customer"


def _route_after_customer_turn(state: PipelineState) -> Literal["agent", "consistency", "cap"]:
    if state.dialogue_done:
        return "consistency"
    if len(state.current_dialogue_turns) >= state.dialogue_turn_cap:
        return "cap"
    return "agent"


def _route_validation_enabled(state: PipelineState) -> Literal["validate", "skip"]:
    return "validate" if state.validation_enabled else "skip"


def _route_after_validate_conversation(
    state: PipelineState,
) -> Literal["pass", "retry", "exhausted"]:
    assert state.last_verdict is not None, "validate_conversation_node must set last_verdict"
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
    adb: AsyncDatabase | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the Phase 2 (turn-based dialogue) subgraph.

    Returns a graph that, given a ``PipelineState`` with ``problems_committed``
    populated by Phase 1, builds a deterministic allocation plan and generates
    one turn-based conversation per slot, persisting incoming requests,
    resolutions, and backfilling pre-recorded lineage rows. No checkpointer is
    attached here — the parent graph owns checkpoint persistence.
    """
    adb = adb if adb is not None else AsyncDatabase(db.path)
    g: StateGraph[PipelineState, Any, PipelineState, PipelineState] = StateGraph(PipelineState)
    g.add_node(
        "build_allocation_plan",
        partial(build_allocation_plan_node, db=db, adb=adb, settings=settings),
    )
    g.add_node(
        "generate_incoming_request",
        partial(generate_incoming_request_node, factory=factory, db=db, adb=adb, settings=settings),
    )
    g.add_node(
        "generate_agent_turn",
        partial(generate_agent_turn_node, factory=factory, db=db, adb=adb, settings=settings),
    )
    g.add_node(
        "generate_customer_turn",
        partial(generate_customer_turn_node, factory=factory, db=db, adb=adb, settings=settings),
    )
    g.add_node(
        "validate_conversation",
        partial(validate_conversation_node, factory=factory, db=db, adb=adb, settings=settings),
    )
    g.add_node("commit_dialogue", partial(commit_dialogue_node, db=db, adb=adb))
    g.add_node("_bump_retry", _bump_retry_node)
    g.add_node("_mark_exhausted", _mark_exhausted_node)
    g.add_node("_mark_validation_skipped", _mark_validation_skipped_node)
    g.add_node("_mark_cap_hit", _mark_cap_hit_node)
    g.add_node("_route_consistency_or_skip", _converge_node)

    g.add_edge(START, "build_allocation_plan")
    g.add_conditional_edges(
        "build_allocation_plan",
        _route_after_build_allocation_plan,
        {"generate": "generate_incoming_request", "done": END},
    )
    g.add_edge("generate_incoming_request", "generate_agent_turn")
    g.add_conditional_edges(
        "generate_agent_turn",
        _route_after_agent_turn,
        {
            "customer": "generate_customer_turn",
            "consistency": "_route_consistency_or_skip",
            "cap": "_mark_cap_hit",
        },
    )
    g.add_conditional_edges(
        "generate_customer_turn",
        _route_after_customer_turn,
        {
            "agent": "generate_agent_turn",
            "consistency": "_route_consistency_or_skip",
            "cap": "_mark_cap_hit",
        },
    )
    g.add_edge("_mark_cap_hit", "_route_consistency_or_skip")
    g.add_conditional_edges(
        "_route_consistency_or_skip",
        _route_validation_enabled,
        {"validate": "validate_conversation", "skip": "_mark_validation_skipped"},
    )
    g.add_edge("_mark_validation_skipped", "commit_dialogue")
    g.add_conditional_edges(
        "validate_conversation",
        _route_after_validate_conversation,
        {
            "pass": "commit_dialogue",
            "retry": "_bump_retry",
            "exhausted": "_mark_exhausted",
        },
    )
    g.add_edge("_bump_retry", "generate_incoming_request")
    g.add_edge("_mark_exhausted", "commit_dialogue")
    g.add_conditional_edges(
        "commit_dialogue",
        _route_after_commit_resolution,
        {"next": "generate_incoming_request", "done": END},
    )

    return g.compile()
