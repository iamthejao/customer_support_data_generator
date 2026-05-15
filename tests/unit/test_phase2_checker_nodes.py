from pathlib import Path

import pytest

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase2_cases.nodes import (
    turn_background_check_node,
    turn_consistency_check_node,
    turn_scenario_check_node,
)
from csfd.phases.phase2_cases.state import TurnDraft
from csfd.prompts.registry import PromptRegistry
from csfd.settings import AgentLLMConfig


def _factory(canned: Verdict) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.1)
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={Verdict: canned}),
        agent_configs={
            "turn_consistency": cfg,
            "turn_background": cfg,
            "turn_scenario": cfg,
        },
    )


@pytest.mark.asyncio
async def test_turn_consistency_check_returns_verdict() -> None:
    canned = Verdict(checker="consistency", passed=True)
    factory = _factory(canned)
    payload = {
        "current_turn_draft": TurnDraft(
            speaker="agent",
            content="hello",
            intent="question",
        ).model_dump()
    }
    update = await turn_consistency_check_node(payload, factory=factory)
    assert update["verdicts"][0].checker == "turn_consistency"


@pytest.mark.asyncio
async def test_turn_background_check_returns_verdict() -> None:
    canned = Verdict(checker="background", passed=False)
    factory = _factory(canned)
    payload = {
        "current_turn_draft": TurnDraft(
            speaker="agent",
            content="hello",
            intent="question",
        ).model_dump()
    }
    update = await turn_background_check_node(payload, factory=factory)
    assert update["verdicts"][0].checker == "turn_background"
    assert update["verdicts"][0].passed is False


@pytest.mark.asyncio
async def test_turn_scenario_check_returns_verdict() -> None:
    canned = Verdict(checker="scenario", passed=True)
    factory = _factory(canned)
    payload = {
        "current_turn_draft": TurnDraft(
            speaker="agent",
            content="hello",
            intent="question",
        ).model_dump()
    }
    update = await turn_scenario_check_node(payload, factory=factory)
    assert update["verdicts"][0].checker == "turn_scenario"
