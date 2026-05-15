"""Tests for CreativeNoise agent emitting TurnModification."""

import jinja2
import pytest

from csfd.agents.base import AgentContext
from csfd.agents.creative_noise import CreativeNoise, TurnModification
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle


def _handle() -> PromptHandle:
    src = "Apply noise: {{ inputs.turn_content }}"
    env = jinja2.Environment(autoescape=False)
    return PromptHandle(name="n", template=env.from_string(src), source=src, version="nnnnnnnnnnnn")


@pytest.mark.asyncio
async def test_creative_noise_returns_modified_turn() -> None:
    canned = TurnModification(
        modified_content="hey help my printer is on fire :(",
        noise_type="typos_informal_phrasing",
        rationale="Customer is panicking",
    )
    llm = FakeChatModel(structured={TurnModification: canned})
    noise = CreativeNoise(name="creative_noise", prompt=_handle(), llm=llm)
    out = await noise.invoke(AgentContext(inputs={"turn_content": "help my printer is on fire"}))
    assert isinstance(out, TurnModification)
    assert out.noise_type == "typos_informal_phrasing"
    assert "fire" in out.modified_content
