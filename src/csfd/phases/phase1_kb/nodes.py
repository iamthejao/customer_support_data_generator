"""Async LangGraph nodes for the Phase 1 KB-generation subgraph."""

from __future__ import annotations

from typing import Any

from csfd.agents.base import AgentContext
from csfd.agents.factory import AgentFactory
from csfd.phases.phase1_kb.state import KBState, ProblemDraft


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
