"""Async LangGraph nodes for the Phase 1 KB-generation subgraph."""

from __future__ import annotations

from typing import Any

from csfd.agents.base import AgentContext
from csfd.agents.factory import AgentFactory
from csfd.phases.phase1_kb.dedup import lexical_dedup
from csfd.phases.phase1_kb.state import (
    CommittedProblem,
    CoverageDecision,
    KBArticleDraft,
    KBState,
    ProblemDraft,
)


async def seed_load_node(state: KBState) -> dict[str, Any]:
    """No-op node: the seeds are already loaded into KBState at run construction.

    Kept as an explicit graph entrypoint for readability and so future variations
    have a stable home.
    """
    return {}


async def problem_brainstorm_node(
    state: KBState,
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    """Generate a candidate Problem via the `problem_brainstorm` Generator agent."""
    gen = factory.build_generator(
        name="problem_brainstorm",
        prompt_name="phase1.problem_generator",
        output_schema_factory=lambda: ProblemDraft,
    )
    ctx = AgentContext(
        inputs={
            "company_name": state.company.name,
            "company_sections": state.company.sections,
            "scenarios": [
                {"category": s.category, "title": s.title, "summary": s.summary}
                for s in state.scenarios.scenarios
            ],
        },
        prior_verdicts=state.verdicts,
        retry_attempt=state.retry_attempt,
    )
    draft = await gen.invoke(ctx)
    if not isinstance(draft, ProblemDraft):
        raise TypeError(f"Expected ProblemDraft, got {type(draft).__name__}")
    return {"current_problem_draft": draft}


def dedup_node(state: KBState, *, similarity_threshold: float) -> dict[str, Any]:
    """Drop near-duplicate committed problems via lexical TF-IDF cosine.

    Identifies kept problems by mapping their drafts back to CommittedProblem entries.
    """
    drafts = [p.draft for p in state.problems_committed]
    kept_drafts = lexical_dedup(drafts, threshold=similarity_threshold)
    kept_ids = {id(d) for d in kept_drafts}
    kept = [p for p in state.problems_committed if id(p.draft) in kept_ids]
    return {"problems_committed": kept}


async def coverage_decider_node(
    state: KBState,
    *,
    factory: AgentFactory,
    target_rate: float,
) -> dict[str, Any]:
    """For each committed problem: invoke coverage_judge to decide has_kb,
    then top-up or trim borderline cases to hit `target_rate`."""
    judge = factory.build_generator(
        name="coverage_judge",
        prompt_name="phase1.problem_generator",  # placeholder; real prompt in Plan 5
        output_schema_factory=lambda: CoverageDecision,
    )
    decided: list[tuple[CommittedProblem, CoverageDecision]] = []
    for p in state.problems_committed:
        ctx = AgentContext(
            inputs={
                "problem": p.draft.model_dump(),
                "company_name": state.company.name,
            }
        )
        decision = await judge.invoke(ctx)
        if not isinstance(decision, CoverageDecision):
            raise TypeError(f"Expected CoverageDecision, got {type(decision).__name__}")
        decided.append((p, decision))

    n = len(decided)
    target_yes = round(n * target_rate)
    current_yes = sum(1 for _, d in decided if d.has_kb)

    if current_yes != target_yes:
        diff = target_yes - current_yes  # positive = need more yes
        flippable = [
            (idx, d)
            for idx, (_, d) in enumerate(decided)
            if (diff > 0 and not d.has_kb) or (diff < 0 and d.has_kb)
        ]
        flippable.sort(key=lambda x: 0 if x[1].confidence == "low" else 1)
        for idx, _ in flippable[: abs(diff)]:
            p, d = decided[idx]
            decided[idx] = (
                p,
                CoverageDecision(
                    has_kb=not d.has_kb,
                    reasoning=f"Adjusted from '{d.reasoning}' to hit target rate",
                    confidence=d.confidence,
                ),
            )

    updated = [
        CommittedProblem(
            id=p.id,
            draft=p.draft,
            has_kb=d.has_kb,
            coverage_reasoning=d.reasoning,
            coverage_confidence=d.confidence,
            quality_flag=p.quality_flag,
            unresolved_issues=p.unresolved_issues,
        )
        for (p, d) in decided
    ]
    return {"problems_committed": updated}


async def article_writer_node(
    state: KBState,
    *,
    factory: AgentFactory,
    problem: CommittedProblem,
) -> dict[str, Any]:
    """Write a KB article for the given covered problem."""
    writer = factory.build_generator(
        name="article_writer",
        prompt_name="phase1.problem_generator",  # placeholder; real prompt in Plan 5
        output_schema_factory=lambda: KBArticleDraft,
    )
    ctx = AgentContext(
        inputs={
            "problem": problem.draft.model_dump(),
            "company_name": state.company.name,
            "company_sections": state.company.sections,
        },
        prior_verdicts=state.verdicts,
        retry_attempt=state.retry_attempt,
    )
    draft = await writer.invoke(ctx)
    if not isinstance(draft, KBArticleDraft):
        raise TypeError(f"Expected KBArticleDraft, got {type(draft).__name__}")
    return {"current_article_draft": draft}
