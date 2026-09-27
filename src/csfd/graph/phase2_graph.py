"""Phase 2 subgraph — plan-driven turn-based dialogue generation.

This module defines the LangGraph subgraph that generates one customer-service
conversation per allocation slot. It first builds a deterministic allocation
plan (via :func:`csfd.allocator.build_allocation_plan`) and pre-records the
lineage rows; then, for each slot, it runs a turn-by-turn dialogue between two
information-asymmetric agents:

* the **customer** agent sees only customer-observable problem fields
  (symptoms, impact, persona), what they find when asked to run each check of
  the problem's diagnosis plan, and the conversation so far;
* the **service** agent is never told the root cause: it sees a troubleshooting
  guide built from the diagnosis plan (candidate causes, checks without their
  results; the fix only on a contact planned to reach the cause), the contact's
  planned beat and ending, and the conversation so far (see :mod:`csfd.diagnosis`).

Each case is planned up front to end in a problem state (see
:mod:`csfd.outcomes`); the consistency check reports the state the dialogue
actually reached, and a miss against the plan fails the attempt.

Both also see the case's seeded facts (see :mod:`csfd.facts`): the customer
what a customer knows (their machine's model and serial, site, job role), the
agent what the CRM shows, so neither has to invent identifiers.

Turn 1 is the customer's opening message (the standalone incoming request).
Turns then alternate, starting with the service agent, until whichever speaker
just spoke flags ``done`` — or a hard ``dialogue.turn_cap`` is reached. With
``tickets.channel == "phone"`` the same loop runs on the ``phone_*`` prompts:
turn 1 is the agent's scripted greeting (see :mod:`csfd.calls`), turn 2 the
caller's opening, and per-utterance timings are estimated at commit.

A slot is one *case*. With ``tickets.rounds`` configured, a case spans several
contacts (see :mod:`csfd.rounds`): after a non-final contact commits, the loop
re-enters ``generate_incoming_request`` for the same slot with the committed
contacts as case history. A planned ``dropped`` contact reuses the turn-cap
route with a lower per-contact cap and ends with ``end_reason="dropped"``. A
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

import dataclasses
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from csfd.agents.base import AgentContext, Issue, Verdict
from csfd.agents.factory import AgentFactory
from csfd.agents.tracing import ParentLink, TracingAdapter
from csfd.allocator import ProblemRef, build_allocation_plan
from csfd.calls import (
    describe_gap,
    estimate_email_times,
    estimate_turn_timings,
    scripted_greeting,
)
from csfd.case_plan import CasePlanEntry
from csfd.diagnosis import (
    ContactBeat,
    DiagnosisPlan,
    agent_guide,
    customer_findings,
    plan_beats,
)
from csfd.facts import (
    CaseFacts,
    crm_view,
    customer_view,
    draw_case_facts,
    identifier_mismatches,
)
from csfd.graph.pipeline_graph import PipelineState, _PlanSlot, _PriorContact, _RoundSpec
from csfd.outcomes import (
    ContactEnding,
    plan_problem_states,
    planned_ending,
)
from csfd.pipeline import (
    ConsistencyVerdict,
    DialogueTurnOutput,
    IncomingRequestOutput,
    ResolutionOutput,
    _apply_consistency_edits,
    _assemble_resolution,
)
from csfd.rounds import assign_round_counts, contact_label, plan_case_rounds
from csfd.settings import AppSettings
from csfd.storage.db import Database
from csfd.storage.db_async import AsyncDatabase
from csfd.storage.repository import (
    IncomingRequestRecord,
    IncomingRequestRepo,
    LineageRecord,
    LineageRepo,
    ProblemRecord,
    ResolutionRecord,
    ResolutionRepo,
)
from csfd.ticket_types.definitions import ProblemComplexity
from csfd.utils.rng import derive_rng

# --------------------------------------------------------------------------- #
# Input builders (enforce information asymmetry at the prompt boundary)
# --------------------------------------------------------------------------- #


def _render_history(turns: list[DialogueTurnOutput]) -> list[dict[str, str]]:
    """Render accumulated turns into the simple {speaker, content} dicts prompts expect."""
    return [{"speaker": t.speaker, "content": t.content} for t in turns]


def _customer_name(slot: _PlanSlot) -> str:
    return f"Customer-{slot.tier}-{slot.index:04d}"


def _agent_name(slot: _PlanSlot) -> str:
    """The one agent who handles this case — every round of a case shares it."""
    return f"Agent-{slot.ticket_type.value}-{slot.index:04d}"


def _current_round(state: PipelineState, slot: _PlanSlot) -> _RoundSpec:
    """The contact being generated; a slot without a round plan is one final contact."""
    return slot.rounds[state.round_index] if slot.rounds else _RoundSpec()


def _contact_key(state: PipelineState, slot: _PlanSlot, rnd: _RoundSpec) -> str:
    """Stable id of one contact: the case's ticket_uid, suffixed for rounds after the first."""
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    return ticket_uid if rnd.sequence == 1 else f"{ticket_uid}:r{rnd.sequence:02d}"


def _effective_turn_cap(state: PipelineState) -> int:
    """The dialogue cap for the current contact: a planned drop cuts it short."""
    cap = state.dialogue_turn_cap
    if state.slot_index < len(state.plan_slots):
        rnd = _current_round(state, state.plan_slots[state.slot_index])
        if rnd.drop_after_turns is not None:
            return min(cap, rnd.drop_after_turns)
    return cap


# How an earlier contact of the same case finished, as the next round's prompts
# describe it. "follow_up" is reserved for a contact that really did agree a next
# step; everything else that ended without one gets its own label.
EndedLabel = Literal["dropped", "cap_hit", "frustrated", "follow_up", "unresolved"]


def _ended_label(contact: _PriorContact) -> EndedLabel:
    """Describe how an already-committed contact of this case ended."""
    if contact.end_reason == "dropped":
        return "dropped"
    if contact.end_reason == "cap_hit":
        return "cap_hit"
    last_reason = contact.turns[-1].done_reason if contact.turns else None
    if last_reason == "customer_frustrated":
        return "frustrated"
    if last_reason == "follow_up":
        return "follow_up"
    return "unresolved"


def _round_inputs(state: PipelineState, rnd: _RoundSpec) -> dict[str, Any]:
    """Round context + earlier contacts for multi-contact cases; empty for single contacts.

    Both speakers see the earlier transcripts: the customer lived through them
    and the agent reads them from the case history, so no hidden fields leak.
    """
    if rnd.count <= 1:
        return {}
    since_previous = None
    if state.case_history and rnd.started_at is not None:
        prev = state.case_history[-1]
        prev_end = prev.ended_at or prev.started_at
        if prev_end is not None:
            since_previous = describe_gap((rnd.started_at - prev_end).total_seconds())
    return {
        "round": {
            "sequence": rnd.sequence,
            "count": rnd.count,
            "end_mode": rnd.end_mode,
            "since_previous": since_previous or "some time",
        },
        "case_history": [
            {
                "sequence": c.sequence,
                "when": c.started_at.strftime("%a %d %b %Y, %H:%M") if c.started_at else "earlier",
                "ended": _ended_label(c),
                "turns": _render_history(c.turns),
            }
            for c in state.case_history
        ],
    }


def _prompt_name(settings: AppSettings, base: str) -> str:
    """Resolve a Phase 2 prompt for the configured channel (``phone_*`` variants)."""
    prefix = "phone_" if settings.tickets.channel == "phone" else ""
    return f"phase2.{prefix}{base}"


# Issues from the code-level identifier check (csfd.facts); they name no root
# cause, so a re-roll can show them to the customer as well as to the agent.
_CASE_FACTS_RULE = "case_facts"


def _fact_issues(state: PipelineState) -> list[str]:
    """Identifier mismatches that failed the previous attempt of this contact."""
    verdict = state.last_verdict
    if verdict is None or verdict.passed:
        return []
    return [i.explanation for i in verdict.issues if i.rule_violated == _CASE_FACTS_RULE]


def _contact_beat(
    state: PipelineState, slot: _PlanSlot, rnd: _RoundSpec
) -> tuple[ContactBeat | None, list[int]]:
    """This contact's beat and the plan checks already run, from how earlier contacts ended.

    Earlier contacts are walked in order with the checks still pending: one that
    ended with its agreed next step ran the pending checks and its own; any other
    (cut off, capped, broken off) leaves its checks pending. The pending checks
    move into this contact, unless it is a planned drop that reports no results.
    """
    if rnd.beat is None:
        return None, []
    ended = {c.sequence: _ended_label(c) for c in state.case_history}
    done: list[int] = []
    pending: list[int] = []
    for r in slot.rounds:
        if r.sequence >= rnd.sequence or r.beat is None:
            continue
        pending.extend(r.beat.checks)
        if ended.get(r.sequence) == "follow_up":
            done.extend(pending)
            pending = []
    if rnd.end_mode == "dropped":
        pending = []
    return rnd.beat.model_copy(update={"checks": [*pending, *rnd.beat.checks]}), done


def _planned_contact(slot: _PlanSlot, rnd: _RoundSpec) -> dict[str, Any] | None:
    """How this contact is planned to end: the state it leaves the case in and the ending."""
    if rnd.beat is None:
        return None
    return {
        "problem_state": rnd.beat.problem_state,
        "ending": planned_ending(rnd.beat.problem_state, rnd.end_mode),
        "cause_confirmed": rnd.beat.cause_confirmed,
        "final": rnd.sequence == rnd.count,
    }


def _agent_plan(
    state: PipelineState, slot: _PlanSlot, rnd: _RoundSpec, beat: ContactBeat | None
) -> dict[str, Any] | None:
    """The contact's plan as the agent sees it: the ending only once the diagnosis is done.

    The agent diagnoses without knowing how the case ends, so the planned state
    cannot point it to a cause. The ending is revealed once an agent turn has
    reported the diagnosis done (``diagnosis_done``), or straight away when this
    contact has no checks to run.
    """
    planned = _planned_contact(slot, rnd)
    if planned is None:
        return None
    revealed = (
        beat is None
        or not beat.checks
        or any(t.diagnosis_done for t in state.current_dialogue_turns if t.speaker == "agent")
    )
    if not revealed:
        return {"final": planned["final"], "revealed": False}
    return {**planned, "revealed": True}


def _agreed_test(
    state: PipelineState, plan: DiagnosisPlan | None, slot: _PlanSlot, rnd: _RoundSpec
) -> dict[str, str] | None:
    """The check the customer agreed to run after the previous contact, with its result.

    None unless the previous contact really ended with that agreement.
    """
    previous = [r for r in slot.rounds if r.sequence == rnd.sequence - 1]
    beat = previous[0].beat if previous else None
    agreed = bool(state.case_history) and _ended_label(state.case_history[-1]) == "follow_up"
    if plan is None or beat is None or beat.next_check is None or not agreed:
        return None
    check = plan.checks[beat.next_check]
    return {"how_to_check": check.how_to_check, "finding": check.finding}


def _customer_inputs(state: PipelineState, slot: _PlanSlot) -> dict[str, Any]:
    """Customer-view inputs: symptoms, impact, persona, own facts and check findings.

    No root cause: the customer only knows what they find when asked to do each
    check of this contact's beat (and what they did in earlier contacts).
    """
    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    rnd = _current_round(state, slot)
    plan = _diagnosis_plan(problem)
    beat, done = _contact_beat(state, slot, rnd)
    return {
        "company_name": state.company.name,
        "ticket_type": slot.ticket_type.value,
        "customer_name": _customer_name(slot),
        "customer_tier": slot.tier,
        "customer_tone": slot.tone,
        "symptoms": problem.symptoms,
        "customer_impact": problem.customer_impact,
        "category": problem.category,
        "facts": customer_view(slot.facts) if slot.facts else None,
        "fact_issues": _fact_issues(state),
        "turn_cap": _effective_turn_cap(state),
        "diagnosis": customer_findings(plan, beat, done_checks=done) if plan and beat else None,
        "agreed_test": _agreed_test(state, plan, slot, rnd),
        "planned": _planned_contact(slot, rnd),
        **_round_inputs(state, rnd),
    }


def _agent_inputs(state: PipelineState, slot: _PlanSlot) -> dict[str, Any]:
    """Service-view inputs: a troubleshooting guide, the CRM record, the planned beat.

    The agent is not told the root cause. It gets the problem's diagnosis plan as
    a guide (every candidate cause with its fix, in a neutral order, and the
    checks without their results) and has to get there through the customer's
    answers.
    """
    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    rnd = _current_round(state, slot)
    plan = _diagnosis_plan(problem)
    beat, done = _contact_beat(state, slot, rnd)
    guide = (
        agent_guide(plan, beat, done_checks=done, seed_label=f"{state.run_seed}:{slot.problem_id}")
        if plan and beat
        else None
    )
    return {
        "company_name": state.company.name,
        "ticket_type": slot.ticket_type.value,
        "agent_name": _agent_name(slot),
        "guide": guide,
        "planned": _agent_plan(state, slot, rnd, beat),
        "facts": crm_view(slot.facts) if slot.facts else None,
        "turn_cap": _effective_turn_cap(state),
        **_round_inputs(state, rnd),
    }


def _problem_text(problem: ProblemRecord) -> str:
    """Every free-text field of a problem, for spotting a catalogue model it names."""
    return "\n".join(
        [problem.title, problem.summary, problem.background, *problem.symptoms, *problem.root_cause]
    )


def _diagnosis_plan(problem: ProblemRecord) -> DiagnosisPlan | None:
    """The problem's canonical diagnosis plan; None for a problem written without one."""
    return DiagnosisPlan.model_validate(problem.diagnosis_plan) if problem.diagnosis_plan else None


def _case_plan(slot: _PlanSlot) -> dict[str, Any]:
    """The case's planned course, stored with its lineage as part of the ground truth."""
    return {
        "problem_state": slot.problem_state,
        "contacts": [
            {
                "sequence": r.sequence,
                "end_mode": r.end_mode,
                "planned_ending": planned_ending(
                    r.beat.problem_state if r.beat else None, r.end_mode
                ),
                **(r.beat.model_dump(mode="json") if r.beat else {}),
            }
            for r in slot.rounds
        ],
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
    """Build the deterministic allocation plan and pre-record lineage rows.

    With a case plan (see csfd.case_plan), the proportional plan is built for
    as many cases as it lists, then each entry's pinned dimensions override it.
    """
    tickets_cfg = settings.tickets
    case_plan = state.case_plan
    plan = build_allocation_plan(
        total=len(case_plan.cases) if case_plan else tickets_cfg.total,
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

    seed = settings.pipeline.run_seed or 0
    problems_by_id = {p.id: p for p in state.problems_committed}
    slots = list(plan.slots)
    pins: dict[int, CasePlanEntry] = {}
    if case_plan is not None:
        problem_ids = [p.id for p in state.problems_committed]
        pins = {slot.index: entry for slot, entry in zip(slots, case_plan.cases, strict=True)}
        slots = [
            dataclasses.replace(
                slot,
                problem_id=case_plan.problem_id(pins[slot.index], problem_ids) or slot.problem_id,
                ticket_type=pins[slot.index].ticket_type or slot.ticket_type,
                tier=pins[slot.index].tier or slot.tier,
                tone=pins[slot.index].tone or slot.tone,
            )
            for slot in slots
        ]
    facts: dict[int, CaseFacts] = {
        s.index: draw_case_facts(
            state.company.case_facts,
            seed=seed,
            slot_index=s.index,
            problem_text=_problem_text(problems_by_id[s.problem_id]),
        )
        for s in slots
    }
    states = plan_problem_states(
        [s.index for s in slots],
        {s.index: problems_by_id[s.problem_id].viable_outcomes for s in slots},
        {str(k): w for k, w in tickets_cfg.outcome_proportions.items()},
        seed=seed,
    )
    round_counts = assign_round_counts(
        [s.index for s in slots], tickets_cfg.rounds.proportions, seed=seed
    )
    for index, entry in pins.items():
        if entry.contacts is not None:
            round_counts[index] = entry.contacts
        if entry.problem_state is not None:
            states[index] = entry.problem_state
    pydantic_slots: list[_PlanSlot] = []
    for s in slots:
        specs = plan_case_rounds(
            slot_index=s.index,
            round_count=round_counts[s.index],
            rounds=tickets_cfg.rounds,
            calendar=tickets_cfg.calendar,
            seed=seed,
            opening_turns=2 if tickets_cfg.channel == "phone" else 1,
            turn_cap=tickets_cfg.dialogue.turn_cap,
            end_modes=pins[s.index].end_modes if s.index in pins else None,
        )
        beats = plan_beats(
            _diagnosis_plan(problems_by_id[s.problem_id]),
            [r.end_mode for r in specs],
            states[s.index],
        )
        pydantic_slots.append(
            _PlanSlot(
                index=s.index,
                problem_id=s.problem_id,
                ticket_type=s.ticket_type,
                tier=s.tier,
                tone=s.tone,
                rounds=[
                    _RoundSpec(
                        sequence=r.sequence,
                        count=r.count,
                        started_at=r.started_at,
                        end_mode=r.end_mode,
                        drop_after_turns=r.drop_after_turns,
                        beat=beat,
                    )
                    for r, beat in zip(specs, beats, strict=True)
                ],
                facts=facts[s.index],
                problem_state=states[s.index],
            )
        )

    lineage_repo = LineageRepo(db)
    for slot in pydantic_slots:
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
                case_facts=facts[slot.index].model_dump(),
                problem_state=str(slot.problem_state) if slot.problem_state else None,
                case_plan=_case_plan(slot),
            ),
        )
    return {
        "plan_slots": pydantic_slots,
        "slot_index": 0,
        "round_index": 0,
        "case_history": [],
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
    """LLM call: customer's opening message. Seeds turn 1 (customer).

    On the phone channel the agent's scripted greeting is prepended as turn 1
    (no LLM call) and the caller's opening becomes turn 2.
    """
    slot = state.plan_slots[state.slot_index]
    rnd = _current_round(state, slot)
    contact_key = _contact_key(state, slot, rnd)
    inputs = _customer_inputs(state, slot)
    opening: list[DialogueTurnOutput] = []
    if settings.tickets.channel == "phone":
        assert rnd.started_at is not None, "build_allocation_plan must schedule the call"
        greeting = scripted_greeting(
            company=state.company.name,
            agent=_agent_name(slot),
            at=rnd.started_at,
            rng=derive_rng(state.run_seed, f"{contact_label(slot.index, rnd.sequence)}:greeting"),
        )
        opening.append(DialogueTurnOutput(speaker="agent", content=greeting, done=False))
        inputs["agent_greeting"] = greeting
        inputs["disfluency"] = settings.tickets.phone.disfluency

    link = ParentLink()
    generator = factory.build_generator(
        name="incoming_request_generator",
        prompt_name=_prompt_name(settings, "incoming_request"),
        output_schema_factory=lambda: IncomingRequestOutput,
    )
    traced = TracingAdapter(
        inner=generator,
        db=db,
        adb=adb,
        run_id=state.run_id,
        node_name="incoming_request_generator",
        artifact_type="resolution",
        artifact_id=contact_key,
        role="generator",
        parent_link=link,
    )
    result = await traced.invoke(AgentContext(inputs=inputs, retry_attempt=state.retry_attempt))
    assert isinstance(result, IncomingRequestOutput)
    opening.append(DialogueTurnOutput(speaker="customer", content=result.body, done=False))
    return {
        "current_resolution_draft": ResolutionOutput(subject=result.subject, body=result.body),
        "current_dialogue_turns": opening,
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
    settings: AppSettings,
    speaker: Literal["customer", "agent"],
) -> dict[str, Any]:
    """Shared body for the customer / agent turn nodes: one LLM call, one appended turn."""
    slot = state.plan_slots[state.slot_index]
    contact_key = _contact_key(state, slot, _current_round(state, slot))
    if speaker == "customer":
        node_name = "customer_turn_generator"
        prompt_name = _prompt_name(settings, "customer_turn")
        inputs = _customer_inputs(state, slot)
    else:
        node_name = "agent_turn_generator"
        prompt_name = _prompt_name(settings, "agent_turn")
        inputs = _agent_inputs(state, slot)
    inputs["conversation_so_far"] = _render_history(state.current_dialogue_turns)
    inputs["turn_index"] = len(state.current_dialogue_turns) + 1
    if settings.tickets.channel == "phone":
        inputs["disfluency"] = settings.tickets.phone.disfluency
    # On a re-roll, surface the prior consistency issues to the agent so it can
    # avoid repeating them. Only the agent turn receives them (it drives diagnosis).
    # Before the ending is revealed, outcome issues would give the planned state away.
    hidden = speaker == "agent" and not (inputs.get("planned") or {}).get("revealed", True)
    prior_issues = (
        [
            i.explanation
            for i in state.last_verdict.issues
            if not (hidden and i.rule_violated == _OUTCOME_RULE)
        ]
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
        artifact_id=contact_key,
        role="generator",
        parent_link=link,
        step=inputs["turn_index"],
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
    return await _generate_turn(
        state, factory=factory, db=db, adb=adb, settings=settings, speaker="agent"
    )


async def generate_customer_turn_node(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
    adb: AsyncDatabase,
    settings: AppSettings,
) -> dict[str, Any]:
    """LLM call: one customer turn."""
    return await _generate_turn(
        state, factory=factory, db=db, adb=adb, settings=settings, speaker="customer"
    )


# Issues from the code-level outcome check: the dialogue did not reach the
# contact's planned problem state or ending.
_OUTCOME_RULE = "planned_outcome"


def _checker_diagnosis(
    state: PipelineState, problem: ProblemRecord, slot: _PlanSlot, rnd: _RoundSpec
) -> dict[str, Any] | None:
    """The full plan plus this contact's beat, for the consistency check."""
    plan = _diagnosis_plan(problem)
    beat, done = _contact_beat(state, slot, rnd)
    if plan is None or beat is None:
        return None
    return {
        "plan": plan.model_dump(mode="json"),
        "confirming_step": plan.confirming_index() + 1,
        "done_steps": [i + 1 for i in done],
        "beat_steps": [i + 1 for i in beat.checks],
        "next_step": beat.next_check + 1 if beat.next_check is not None else None,
    }


def _outcome_issues(draft: ResolutionOutput, rnd: _RoundSpec) -> list[str]:
    """Where the checked dialogue departs from the contact's planned state and ending."""
    # A capped contact is committed with its own warning flag rather than re-rolled.
    if rnd.beat is None or draft.end_reason == "cap_hit":
        return []
    issues: list[str] = []
    planned_state = rnd.beat.problem_state
    if planned_state is not None and draft.problem_state != planned_state:
        issues.append(
            f"The contact must leave the problem {planned_state.value!r}, but the dialogue "
            f"leaves it {draft.problem_state.value if draft.problem_state else 'undetermined'!r}."
        )
    expected = planned_ending(planned_state, rnd.end_mode)
    # A cap hit is already flagged as a warning; it is not a planned-outcome failure.
    if draft.contact_ending not in (expected, ContactEnding.CAP_HIT):
        got = draft.contact_ending.value if draft.contact_ending else "unknown"
        issues.append(f"The contact must end {expected.value!r}, but it ends {got!r}.")
    return issues


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
    rnd = _current_round(state, slot)
    contact_key = _contact_key(state, slot, rnd)
    # _converge_node has already assembled the accumulated turns into the draft.
    assert state.current_resolution_draft is not None
    draft = state.current_resolution_draft

    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    inputs: dict[str, Any] = {
        "company_name": state.company.name,
        "channel": settings.tickets.channel,
        "customer_tone": slot.tone,
        "problem": {
            "title": problem.title,
            "complexity": problem.complexity,
            "background": problem.background,
            "symptoms": problem.symptoms,
            "root_cause": problem.root_cause,
            "resolution_hint": problem.resolution_hints.get(slot.ticket_type.value, ""),
        },
        "facts": slot.facts.model_dump() if slot.facts else None,
        "diagnosis": _checker_diagnosis(state, problem, slot, rnd),
        "planned": _planned_contact(slot, rnd),
        "candidate": draft.model_dump(),
        **_round_inputs(state, rnd),
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
        artifact_id=contact_key,
        role="checker",
        parent_link=link,
    )
    verdict = await traced.invoke(AgentContext(inputs=inputs, retry_attempt=state.retry_attempt))
    assert isinstance(verdict, ConsistencyVerdict)

    llm_passed = verdict.status in ("pass", "pass_with_edits")
    final_draft = _apply_consistency_edits(draft, verdict) if llm_passed else draft
    # Code-level backstop on the text that would be committed: a model or serial
    # number that contradicts the case record fails the attempt like the checker.
    mismatches = (
        identifier_mismatches(
            [final_draft.subject, *(t.content for t in final_draft.turns)],
            slot.facts,
            state.company.case_facts,
        )
        if slot.facts
        else []
    )
    # The contact must leave the case in its planned state and end the planned way.
    outcome_issues = _outcome_issues(final_draft, rnd) if llm_passed else []
    passed = llm_passed and not mismatches and not outcome_issues
    edited = passed and verdict.status == "pass_with_edits"
    quality_flag = state.last_quality_flag
    # An edited transcript is flagged unless a turn-cap-hit already claimed the
    # slot — the cap is the more actionable signal and must not be overwritten.
    if edited and quality_flag != "warning:turn_cap_hit":
        quality_flag = "info:consistency_edited"
    # Surface the issues on a fail so the re-roll's turns can see them via
    # AgentContext (see _generate_turn's prior_issues and _fact_issues).
    llm_issues = [] if llm_passed else verdict.issues
    issues = (
        []
        if passed
        else [
            Issue(severity="error", location="conversation", rule_violated=rule, explanation=msg)
            for rule, msgs in (
                ("consistency", llm_issues),
                (_CASE_FACTS_RULE, mismatches),
                (_OUTCOME_RULE, outcome_issues),
            )
            for msg in msgs
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


def _timed_turns(
    state: PipelineState, slot: _PlanSlot, rnd: _RoundSpec, turns: list[DialogueTurnOutput]
) -> tuple[list[dict[str, Any]], float]:
    """Attach estimated per-utterance offsets to phone turns; return (turns, duration_s).

    Timing is computed here, after any consistency edits, and never lives on
    ``DialogueTurnOutput`` so the LLM is not asked to produce it.
    """
    rng = derive_rng(state.run_seed, f"{contact_label(slot.index, rnd.sequence)}:timing")
    timings = estimate_turn_timings([t.content for t in turns], rng)
    out: list[dict[str, Any]] = []
    for turn, timing in zip(turns, timings, strict=True):
        row = {**turn.model_dump(), "start_s": timing.start_s, "end_s": timing.end_s}
        if timing.hold_s:
            row["hold_s"] = timing.hold_s
        out.append(row)
    return out, max((t.end_s for t in timings), default=0.0)


def _dated_emails(
    state: PipelineState,
    slot: _PlanSlot,
    rnd: _RoundSpec,
    turns: list[DialogueTurnOutput],
    settings: AppSettings,
) -> tuple[list[dict[str, Any]], datetime]:
    """Attach a seeded ``sent_at`` to each email of a thread; return (turns, last sent_at).

    A thread with a later contact planned in the same case is compressed to end
    before that contact starts.
    """
    assert rnd.started_at is not None
    next_round = slot.rounds[state.round_index + 1] if rnd.sequence < rnd.count else None
    times = estimate_email_times(
        [t.speaker for t in turns],
        started_at=rnd.started_at,
        calendar=settings.tickets.calendar,
        rng=derive_rng(state.run_seed, f"{contact_label(slot.index, rnd.sequence)}:timing"),
        next_contact_at=next_round.started_at if next_round else None,
    )
    out = [
        {**turn.model_dump(), "sent_at": at.isoformat()}
        for turn, at in zip(turns, times, strict=True)
    ]
    return out, times[-1]


async def commit_dialogue_node(
    state: PipelineState,
    *,
    db: Database,
    adb: AsyncDatabase,
    settings: AppSettings,
) -> dict[str, Any]:
    """Persist one contact (incoming request + resolution) and advance the loop.

    Lineage is backfilled from the case's first contact, so the gold-tuple join
    keeps pairing a request with its own resolution; every contact carries the
    case's ``case_uid``. After a non-final contact the same slot continues with
    the next round; after the last one the loop moves to the next slot.
    """
    slot = state.plan_slots[state.slot_index]
    rnd = _current_round(state, slot)
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    contact_key = _contact_key(state, slot, rnd)
    assert state.current_resolution_draft is not None
    draft = state.current_resolution_draft

    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    customer_name = _customer_name(slot)
    channel = settings.tickets.channel
    turns: list[dict[str, Any]] = [t.model_dump() for t in draft.turns]
    ended_at: datetime | None = None
    duration_s: float | None = None
    if channel == "phone":
        turns, duration_s = _timed_turns(state, slot, rnd, draft.turns)
        if rnd.started_at is not None:
            ended_at = rnd.started_at + timedelta(seconds=duration_s)
    elif rnd.started_at is not None and draft.turns:
        turns, ended_at = _dated_emails(state, slot, rnd, draft.turns, settings)
        duration_s = (ended_at - rnd.started_at).total_seconds()

    ir_id = await IncomingRequestRepo(db).acreate(
        adb,
        IncomingRequestRecord(
            request_uid=f"{contact_key}:req",
            run_id=state.run_id,
            problem_id=problem.id,
            ticket_type=slot.ticket_type.value,
            customer_name=customer_name,
            customer_tier=slot.tier,
            customer_tone=slot.tone,
            channel=channel,
            subject=draft.subject,
            body=draft.body,
            quality_flag=state.last_quality_flag,
            created_at=datetime.now(UTC),
            case_uid=ticket_uid,
            round_index=rnd.sequence,
        ),
    )
    res_id = await ResolutionRepo(db).acreate(
        adb,
        ResolutionRecord(
            resolution_uid=f"{contact_key}:res",
            run_id=state.run_id,
            incoming_request_id=ir_id,
            problem_id=problem.id,
            ticket_type=slot.ticket_type.value,
            turns=turns,
            turn_count=len(draft.turns),
            # A case only closes on its last contact. The prompts ask for this, but a
            # speaker can still claim done, and an exhausted-retry commit keeps that
            # draft, so the rule is enforced here instead of trusted to the model.
            resolved=draft.resolved and rnd.sequence == rnd.count,
            problem_state=str(draft.problem_state) if draft.problem_state else None,
            planned_problem_state=(
                str(rnd.beat.problem_state) if rnd.beat and rnd.beat.problem_state else None
            ),
            contact_ending=str(draft.contact_ending) if draft.contact_ending else None,
            planned_contact_ending=(
                str(planned_ending(rnd.beat.problem_state, rnd.end_mode)) if rnd.beat else None
            ),
            commitments=[c.model_dump() for c in draft.commitments],
            quality_flag=state.last_quality_flag,
            created_at=datetime.now(UTC),
            channel=channel,
            agent_name=_agent_name(slot),
            end_reason=draft.end_reason,
            started_at=rnd.started_at,
            ended_at=ended_at,
            duration_s=duration_s,
            case_uid=ticket_uid,
            round_index=rnd.sequence,
            round_count=rnd.count,
        ),
    )
    if rnd.sequence == 1:
        await LineageRepo(db).aupdate_links(
            adb, ticket_uid, incoming_request_id=ir_id, resolution_id=res_id
        )

    if rnd.sequence < rnd.count:
        progress: dict[str, Any] = {
            "round_index": state.round_index + 1,
            "case_history": [
                *state.case_history,
                _PriorContact(
                    sequence=rnd.sequence,
                    started_at=rnd.started_at,
                    ended_at=ended_at,
                    end_reason=draft.end_reason,
                    turns=draft.turns,
                ),
            ],
        }
    else:
        progress = {"slot_index": state.slot_index + 1, "round_index": 0, "case_history": []}
    return {
        **progress,
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

    The contact is stored with its planned problem state (unchecked). A
    turn-cap-hit flag takes precedence: a capped conversation that also skips
    validation should still surface the cap, the more actionable signal.
    """
    if state.last_quality_flag == "warning:turn_cap_hit":
        return {}
    # Nothing judged the text, so the contact keeps its planned state, and the
    # flag says it is unchecked.
    update: dict[str, Any] = {"last_quality_flag": "warning:validation_skipped"}
    draft = state.current_resolution_draft
    slots = state.plan_slots
    beat = (
        _current_round(state, slots[state.slot_index]).beat
        if state.slot_index < len(slots)
        else None
    )
    if draft is not None and beat is not None:
        update["current_resolution_draft"] = draft.model_copy(
            update={"problem_state": beat.problem_state}
        )
    return update


async def _mark_cap_hit_node(state: PipelineState) -> dict[str, Any]:
    """Stamp the turn-cap warning and pin the end reason when the cap is reached.

    A contact planned to drop hits its own lower cap first; that is the
    intended ending, so it is recorded as ``dropped`` without a warning.
    """
    if state.slot_index < len(state.plan_slots):
        rnd = _current_round(state, state.plan_slots[state.slot_index])
        if rnd.drop_after_turns is not None and len(state.current_dialogue_turns) >= min(
            rnd.drop_after_turns, state.dialogue_turn_cap
        ):
            return {"dialogue_end_reason": "dropped"}
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
    if len(state.current_dialogue_turns) >= _effective_turn_cap(state):
        return "cap"
    return "customer"


def _route_after_customer_turn(state: PipelineState) -> Literal["agent", "consistency", "cap"]:
    if state.dialogue_done:
        return "consistency"
    if len(state.current_dialogue_turns) >= _effective_turn_cap(state):
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
    """``next`` covers both the next contact of the same case and the next slot."""
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
    g.add_node("commit_dialogue", partial(commit_dialogue_node, db=db, adb=adb, settings=settings))
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
