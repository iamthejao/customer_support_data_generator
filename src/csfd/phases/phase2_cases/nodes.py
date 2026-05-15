"""Async LangGraph nodes for the Phase 2 case-generation subgraph."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from csfd.agents.base import AgentContext
from csfd.agents.creative_noise import TurnModification
from csfd.agents.factory import AgentFactory
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBArticleDraft,
    ProblemDraft,
)
from csfd.phases.phase2_cases.sampler import sample_ticket_count, sample_ticket_type
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CommittedTurn,
    CustomerPersona,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.settings import Phase2Config
from csfd.storage.db import Database
from csfd.storage.repository import (
    KBArticleRepo,
    ProblemRepo,
    TicketRecord,
    TicketRepo,
    TurnRecord,
    TurnRepo,
)
from csfd.ticket_types.definitions import TICKET_TYPE_METADATA, TicketType
from csfd.utils.rng import derive_rng, weighted_choice


async def load_kb_node(state: TicketState, *, db: Database) -> dict[str, Any]:
    """Load problems + KB articles from the parent Phase 1 run."""
    problems = ProblemRepo(db).list_for_run(state.parent_run_id)
    pool: list[CommittedProblem] = []
    for p in problems:
        pool.append(
            CommittedProblem(
                id=p.id,
                draft=ProblemDraft(
                    title=p.title,
                    description=p.description,
                    category=p.category,
                    severity=p.severity,
                    metadata=(json.loads(p.metadata_json) if p.metadata_json else {}),
                ),
                has_kb=p.has_kb,
                coverage_reasoning=p.coverage_reasoning or "",
                coverage_confidence=p.coverage_confidence or "low",
                quality_flag=p.quality_flag,
            )
        )

    kb_by_problem: dict[str, CommittedArticle] = {}
    repo = KBArticleRepo(db)
    for cp in pool:
        if not cp.has_kb:
            continue
        row = repo.get_by_problem(cp.id)
        if row is None:
            continue
        kb_by_problem[cp.id] = CommittedArticle(
            id=row.id,
            problem_id=row.problem_id,
            draft=KBArticleDraft(
                title=row.title,
                content_markdown=row.content_markdown,
                troubleshooting_steps=(
                    json.loads(row.troubleshooting_steps_json)
                    if row.troubleshooting_steps_json
                    else []
                ),
                prerequisites=(
                    json.loads(row.prerequisites_json) if row.prerequisites_json else []
                ),
                metadata=(json.loads(row.metadata_json) if row.metadata_json else {}),
            ),
            quality_flag=row.quality_flag,
        )

    return {"problem_pool": pool, "kb_by_problem": kb_by_problem}


async def ticket_sampler_node(
    state: TicketState,
    *,
    cfg: Phase2Config,
) -> dict[str, Any]:
    """Sample the next (problem, ticket_type) pair for this run.

    Honours the hard `has_kb=False ⇒ L3` rule via `sample_ticket_type`.
    For Plan 4 the sampler picks the *first* problem in the pool that still has
    capacity (Plan 5 will wire the outer loop via a counter on state).
    """
    if not state.problem_pool:
        raise ValueError("ticket_sampler_node called with empty problem_pool")
    rng = derive_rng(state.run_seed, f"ticket_sampler:{len(state.turns_committed)}")
    problem = state.problem_pool[0]
    tt = sample_ticket_type(
        problem,
        rng,
        weights_has_kb=cfg.ticket_type_weights.has_kb,
        weights_no_kb=cfg.ticket_type_weights.no_kb,
    )
    return {
        "current_problem_id": problem.id,
        "ticket_type": tt.value,
    }


async def ticket_init_node(
    state: TicketState,
    *,
    problem: CommittedProblem,
    ticket_type: str,
    kb_article_id: str | None,
) -> dict[str, Any]:
    """Build a `CommittedTicket` (draft form) with synthesised personas."""
    tt = TicketType(ticket_type)
    md = TICKET_TYPE_METADATA[tt]
    rng = derive_rng(state.run_seed, f"persona:{problem.id}:{ticket_type}")
    customer_tier = rng.choice(["standard", "premium", "enterprise"])
    customer_tone = rng.choice(["neutral", "frustrated", "polite", "urgent"])
    ticket = CommittedTicket(
        id=problem.id + ":" + ticket_type + ":" + str(rng.randint(0, 1_000_000)),
        draft=TicketDraft(
            problem_id=problem.id,
            kb_article_id=kb_article_id,
            ticket_type=tt.value,
            priority=("high" if tt == TicketType.L3 else "medium"),
            subject=problem.draft.title,
            customer_persona=CustomerPersona(
                name=f"Customer-{rng.randint(1000, 9999)}",
                tier=customer_tier,
                tone=customer_tone,
            ),
            agent_persona=AgentPersona(
                name=f"Agent-{rng.randint(1000, 9999)}",
                tier=tt.value,
                expertise=md.persona_label,
            ),
        ),
    )
    return {"current_ticket": ticket, "turn_index": 0}


def pick_ticket_count(
    state: TicketState,
    problem: CommittedProblem,
    cfg: Phase2Config,
) -> int:
    """Sample a target ticket count for this problem (helper, not a graph node)."""
    rng = derive_rng(state.run_seed, f"ticket_count:{problem.id}")
    range_inclusive = (
        cfg.tickets_per_problem.has_kb if problem.has_kb else cfg.tickets_per_problem.no_kb
    )
    return sample_ticket_count(rng, range_inclusive=range_inclusive)


async def turn_writer_node(
    state: TicketState,
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    """Generate the next turn for `state.current_ticket`.

    Reads the ticket's persona + prior turns + KB article (if any) so the
    Generator can produce a context-aware next turn.
    """
    if state.current_ticket is None:
        raise ValueError("turn_writer_node called without current_ticket")
    ticket = state.current_ticket
    pid = ticket.draft.problem_id
    article = state.kb_by_problem.get(pid)

    writer = factory.build_generator(
        name="turn_writer",
        prompt_name="phase2.turn_generator",
        output_schema_factory=lambda: TurnDraft,
    )
    ctx = AgentContext(
        inputs={
            "company_name": state.company.name,
            "ticket": ticket.draft.model_dump(),
            "kb_article": article.draft.model_dump() if article else None,
            "prior_turns": [
                {"speaker": t.draft.speaker, "content": t.draft.content, "intent": t.draft.intent}
                for t in state.turns_committed
            ],
            "turn_index": state.turn_index,
        },
        prior_verdicts=state.verdicts,
        retry_attempt=state.retry_attempt,
    )
    draft = await writer.invoke(ctx)
    if not isinstance(draft, TurnDraft):
        raise TypeError(f"Expected TurnDraft, got {type(draft).__name__}")
    return {"current_turn_draft": draft}


def creative_noise_gate(state: TicketState, *, probability: float) -> str:
    """Deterministic per-turn coin flip against `probability`.

    Returns `"apply_noise"` if the seeded RNG draw is below `probability`,
    otherwise `"skip_noise"`. The label is derived from `(run_seed, turn_index,
    current_ticket.id)` so each gate decision is uniquely seeded but
    reproducible.
    """
    if state.current_ticket is None:
        return "skip_noise"
    label = f"noise_gate:{state.current_ticket.id}:{state.turn_index}"
    rng = derive_rng(state.run_seed, label)
    draw = rng.random()
    return "apply_noise" if draw < probability else "skip_noise"


async def creative_noise_node(
    state: TicketState,
    *,
    factory: AgentFactory,
    noise_type_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Apply CreativeNoise to the current turn draft.

    Samples a noise type from `noise_type_weights`, invokes the `creative_noise`
    agent for the modified content, and returns a state update that overwrites
    `current_turn_draft` with the modified version and flips `noise_applied` /
    `noise_type` so persistence records the noise category.
    """
    if state.current_turn_draft is None:
        raise ValueError("creative_noise_node called without current_turn_draft")
    label = (
        f"noise_type:{state.current_ticket.id if state.current_ticket else 'x'}:{state.turn_index}"
    )
    rng = derive_rng(state.run_seed, label)
    noise_type = weighted_choice(noise_type_weights, rng)

    noise_agent = factory.build_creative_noise(
        name="creative_noise",
        prompt_name="phase2.creative_noise",
    )
    ctx = AgentContext(
        inputs={
            "turn_draft": state.current_turn_draft.model_dump(),
            "noise_type": noise_type,
        }
    )
    mod = await noise_agent.invoke(ctx)
    if not isinstance(mod, TurnModification):
        raise TypeError(f"Expected TurnModification, got {type(mod).__name__}")
    modified = state.current_turn_draft.model_copy(
        update={
            "content": mod.modified_content,
            "noise_applied": True,
            "noise_type": mod.noise_type or noise_type,
        }
    )
    return {
        "current_turn_draft": modified,
        "noise_applied": True,
        "noise_type": mod.noise_type or noise_type,
    }


async def _run_turn_checker(
    *,
    factory: AgentFactory,
    name: str,
    prompt_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Helper to run a turn-level checker agent and return verdicts."""
    checker = factory.build_checker(name=name, prompt_name=prompt_name)
    ctx = AgentContext(inputs=payload)
    verdict = await checker.invoke(ctx)
    return {"verdicts": [verdict]}


async def turn_consistency_check_node(
    payload: dict[str, Any],
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    """Check a TurnDraft for consistency via the `turn_consistency` Checker agent."""
    return await _run_turn_checker(
        factory=factory,
        name="turn_consistency",
        prompt_name="phase2.turn_consistency",
        payload=payload,
    )


async def turn_background_check_node(
    payload: dict[str, Any],
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    """Check a TurnDraft for background coherence via the `turn_background` Checker agent."""
    return await _run_turn_checker(
        factory=factory,
        name="turn_background",
        prompt_name="phase2.turn_consistency",  # placeholder; real prompt in Plan 5
        payload=payload,
    )


async def turn_scenario_check_node(
    payload: dict[str, Any],
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    """Check a TurnDraft for scenario relevance via the `turn_scenario` Checker agent."""
    return await _run_turn_checker(
        factory=factory,
        name="turn_scenario",
        prompt_name="phase2.turn_consistency",  # placeholder; real prompt in Plan 5
        payload=payload,
    )


def persist_ticket_node(
    state: TicketState,
    *,
    db: Database,
    ticket: CommittedTicket,
) -> dict[str, Any]:
    """Idempotently persist a committed ticket row."""
    repo = TicketRepo(db)
    repo.create(
        TicketRecord(
            id=ticket.id,
            run_id=state.run_id,
            problem_id=ticket.draft.problem_id,
            kb_article_id=ticket.draft.kb_article_id,
            ticket_type=ticket.draft.ticket_type,
            priority=ticket.draft.priority,
            status=ticket.status,
            subject=ticket.draft.subject,
            customer_persona_json=ticket.draft.customer_persona.model_dump_json(),
            agent_persona_json=ticket.draft.agent_persona.model_dump_json(),
            ground_truth_json=(json.dumps(ticket.ground_truth) if ticket.ground_truth else None),
            metadata_json=(json.dumps(ticket.draft.metadata) if ticket.draft.metadata else None),
            quality_flag=ticket.quality_flag,
            unresolved_issues_json=(
                json.dumps([v.model_dump() for v in ticket.unresolved_issues])
                if ticket.unresolved_issues
                else None
            ),
            created_at=datetime.now(UTC),
            resolved_at=None,
        )
    )
    return {
        "stats": state.stats.model_copy(
            update={
                "tickets_committed": state.stats.tickets_committed + 1,
            }
        )
    }


def persist_turn_node(
    state: TicketState,
    *,
    db: Database,
    turn: CommittedTurn,
) -> dict[str, Any]:
    """Idempotently persist a committed turn row."""
    repo = TurnRepo(db)
    repo.create(
        TurnRecord(
            id=turn.id,
            ticket_id=turn.ticket_id,
            turn_index=turn.turn_index,
            speaker=turn.draft.speaker,
            speaker_persona=turn.draft.speaker_persona,
            content=turn.draft.content,
            intent=turn.draft.intent,
            kb_references_json=(
                json.dumps(turn.draft.kb_references) if turn.draft.kb_references else None
            ),
            noise_applied=turn.draft.noise_applied,
            noise_type=turn.draft.noise_type,
            quality_flag=turn.quality_flag,
            created_at=datetime.now(UTC),
        )
    )
    return {
        "stats": state.stats.model_copy(
            update={
                "turns_committed": state.stats.turns_committed + 1,
                "turns_with_noise": state.stats.turns_with_noise
                + (1 if turn.draft.noise_applied else 0),
            }
        ),
    }
