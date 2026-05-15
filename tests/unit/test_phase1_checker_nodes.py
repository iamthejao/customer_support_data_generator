from pathlib import Path

import pytest

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import problem_consistency_check_node
from csfd.phases.phase1_kb.state import ProblemDraft
from csfd.prompts.registry import PromptRegistry
from csfd.settings import AgentLLMConfig


def _factory(canned: Verdict) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={Verdict: canned}),
        agent_configs={
            "problem_consistency": AgentLLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5",
                temperature=0.1,
            ),
        },
    )


@pytest.mark.asyncio
async def test_problem_consistency_check_returns_verdict_in_state() -> None:
    canned = Verdict(checker="problem_consistency", passed=True)
    factory = _factory(canned)
    payload = {
        "current_problem_draft": ProblemDraft(
            title="t",
            description="d",
            category="c",
            severity="low",
        ),
    }
    update = await problem_consistency_check_node(payload, factory=factory)
    assert "verdicts" in update
    assert len(update["verdicts"]) == 1
    assert update["verdicts"][0].checker == "problem_consistency"
