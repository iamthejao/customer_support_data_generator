from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import coverage_decider_node, dedup_node
from csfd.phases.phase1_kb.state import (
    CommittedProblem,
    CoverageDecision,
    KBState,
    ProblemDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.settings import AgentLLMConfig


def _state(problems: list[CommittedProblem]) -> KBState:
    return KBState(
        run_id=str(uuid4()),
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        scenarios=ScenarioCatalogue(),
        problems_committed=problems,
    )


def _factory(canned: CoverageDecision) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={CoverageDecision: canned}),
        agent_configs={
            "coverage_judge": AgentLLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5",
                temperature=0.2,
            ),
        },
    )


def test_dedup_node_removes_near_duplicates() -> None:
    pcs = [
        CommittedProblem(
            id=str(i),
            draft=ProblemDraft(
                title=f"Login fails {i}",
                description="401 error",
                category="auth",
                severity="low",
            ),
            has_kb=False,
            coverage_reasoning="",
            coverage_confidence="low",
        )
        for i in range(2)
    ] + [
        CommittedProblem(
            id="3",
            draft=ProblemDraft(
                title="Billing overage",
                description="Charged more",
                category="billing",
                severity="medium",
            ),
            has_kb=False,
            coverage_reasoning="",
            coverage_confidence="low",
        ),
    ]
    state = _state(pcs)
    update = dedup_node(state, similarity_threshold=0.69)
    assert "problems_committed" in update
    deduped = update["problems_committed"]
    titles = sorted(p.draft.title for p in deduped)
    assert "Billing overage" in titles
    # The two near-duplicate logins collapse to one
    login_count = sum(1 for p in deduped if "Login" in p.draft.title)
    assert login_count == 1


@pytest.mark.asyncio
async def test_coverage_decider_uses_judge_then_balances_to_target() -> None:
    pcs = [
        CommittedProblem(
            id=str(i),
            draft=ProblemDraft(
                title=f"P{i}",
                description=f"d{i}",
                category="c",
                severity="low",
            ),
            has_kb=False,
            coverage_reasoning="",
            coverage_confidence="low",
        )
        for i in range(10)
    ]
    state = _state(pcs)
    canned = CoverageDecision(has_kb=True, reasoning="ok", confidence="medium")
    factory = _factory(canned)
    update = await coverage_decider_node(
        state,
        factory=factory,
        target_rate=0.7,
    )
    out = update["problems_committed"]
    has_kb_count = sum(1 for p in out if p.has_kb)
    # 70% of 10 = 7
    assert has_kb_count == 7
