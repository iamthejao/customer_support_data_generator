import jinja2
import pytest
from csfd.agents.base import AgentContext, Issue, Verdict
from csfd.agents.generator import Generator
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle
from pydantic import BaseModel


class _Artifact(BaseModel):
    summary: str


def _handle(
    source: str = (
        "Generate something. {{ inputs.x | default('') }}"
        "{% for v in prior_verdicts %}{% for i in v.issues %}"
        "- {{ i.explanation }}"
        "{% endfor %}{% endfor %}"
    ),
) -> PromptHandle:
    env = jinja2.Environment(autoescape=False)
    return PromptHandle(
        name="t",
        template=env.from_string(source),
        source=source,
        version="abc123abc123",
    )


@pytest.mark.asyncio
async def test_generator_returns_typed_artifact() -> None:
    canned = _Artifact(summary="ok")
    llm = FakeChatModel(structured={_Artifact: canned})
    gen = Generator(name="g", prompt=_handle(), output_schema=_Artifact, llm=llm)
    out = await gen.invoke(AgentContext(inputs={"x": "hello"}))
    assert isinstance(out, _Artifact)
    assert out.summary == "ok"


@pytest.mark.asyncio
async def test_generator_renders_prior_verdicts_into_prompt() -> None:
    canned = _Artifact(summary="ok2")
    llm = FakeChatModel(structured={_Artifact: canned})
    gen = Generator(name="g", prompt=_handle(), output_schema=_Artifact, llm=llm)
    verdict = Verdict(
        checker="consistency",
        passed=False,
        issues=[
            Issue(
                severity="error",
                location="x",
                rule_violated="r",
                explanation="Internal id leaked",
            )
        ],
    )
    await gen.invoke(AgentContext(inputs={"x": "hello"}, prior_verdicts=[verdict]))
    rendered = gen.last_rendered_prompt or ""
    assert "Internal id leaked" in rendered
