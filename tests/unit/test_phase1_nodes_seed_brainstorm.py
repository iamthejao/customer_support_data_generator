from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import problem_brainstorm_node
from csfd.phases.phase1_kb.state import KBState, ProblemDraft
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import parse_company_seed
from csfd.seeds.scenarios import parse_scenarios_seed
from csfd.settings import AgentLLMConfig


def _state() -> KBState:
    return KBState(
        run_id=str(uuid4()),
        run_seed=1,
        company=parse_company_seed(Path("tests/fixtures/tiny_company_seed.md")),
        scenarios=parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md")),
    )


def _factory(canned_problem: ProblemDraft) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={ProblemDraft: canned_problem}),
        agent_configs={
            "problem_brainstorm": AgentLLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5",
                temperature=0.5,
            ),
        },
    )


@pytest.mark.asyncio
async def test_problem_brainstorm_emits_draft_into_state() -> None:
    canned = ProblemDraft(
        title="Cannot reset password",
        description="Customer locked out after failed attempts",
        category="Authentication",
        severity="medium",
    )
    factory = _factory(canned)
    state = _state()
    update = await problem_brainstorm_node(state, factory=factory)
    assert "current_problem_draft" in update
    draft = update["current_problem_draft"]
    assert isinstance(draft, ProblemDraft)
    assert draft.title == "Cannot reset password"
