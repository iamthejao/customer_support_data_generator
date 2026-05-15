import jinja2
import pytest
from csfd.agents.base import AgentContext, Verdict
from csfd.agents.checker import Checker
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle


def _handle() -> PromptHandle:
    src = "Inspect: {{ inputs.artifact }}"
    env = jinja2.Environment(autoescape=False)
    return PromptHandle(name="c", template=env.from_string(src), source=src, version="cccccccccccc")


@pytest.mark.asyncio
async def test_checker_emits_passing_verdict() -> None:
    canned = Verdict(checker="consistency", passed=True)
    llm = FakeChatModel(structured={Verdict: canned})
    checker = Checker(name="consistency", prompt=_handle(), llm=llm)
    out = await checker.invoke(AgentContext(inputs={"artifact": "x"}))
    assert isinstance(out, Verdict)
    assert out.passed is True


@pytest.mark.asyncio
async def test_checker_overrides_checker_name_to_match_instance() -> None:
    canned = Verdict(checker="WRONG", passed=False)
    llm = FakeChatModel(structured={Verdict: canned})
    checker = Checker(name="background", prompt=_handle(), llm=llm)
    out = await checker.invoke(AgentContext(inputs={"artifact": "x"}))
    assert isinstance(out, Verdict)
    assert out.checker == "background"
