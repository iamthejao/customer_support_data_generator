import jinja2
import pytest
from pydantic import BaseModel

from csfd.agents.base import AgentContext, Issue, Verdict
from csfd.agents.generator import Generator
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle


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
    gen = Generator(
        name="g",
        prompt=_handle(),
        output_schema=_Artifact,
        llm=llm,
        provider="anthropic",
        model_id="fake",
    )
    out = await gen.invoke(AgentContext(inputs={"x": "hello"}))
    assert isinstance(out, _Artifact)
    assert out.summary == "ok"


@pytest.mark.asyncio
async def test_generator_tracks_token_usage_from_callback() -> None:
    """If the inner LLM emits usage metadata via the callback, Generator exposes it."""
    from typing import Any, ClassVar

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    from langchain_core.runnables import Runnable, RunnableLambda

    canned = _Artifact(summary="ok")

    class UsageFake(FakeChatModel):
        model_config: ClassVar = {"arbitrary_types_allowed": True}

        def with_structured_output(  # type: ignore[override]
            self, schema: type[BaseModel], **kw: Any
        ) -> Runnable[Any, BaseModel]:
            async def _ainvoke(
                _inputs: Any, config: dict[str, Any] | None = None, **__: Any
            ) -> BaseModel:
                msg = AIMessage(
                    content="",
                    usage_metadata={
                        "input_tokens": 12,
                        "output_tokens": 7,
                        "total_tokens": 19,
                    },
                    response_metadata={"model_name": "fake-usage"},
                )
                cb_arg = (config or {}).get("callbacks")
                if cb_arg is None:
                    handlers = []
                elif hasattr(cb_arg, "handlers"):
                    handlers = cb_arg.handlers
                else:
                    handlers = list(cb_arg)
                for cb in handlers:
                    if hasattr(cb, "on_llm_end"):
                        cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
                return canned

            return RunnableLambda(_ainvoke)

    gen = Generator(
        name="g",
        prompt=_handle(),
        output_schema=_Artifact,
        llm=UsageFake(),
        provider="anthropic",
        model_id="fake-usage",
    )
    out = await gen.invoke(AgentContext(inputs={"x": "hi"}))
    assert isinstance(out, _Artifact)
    assert gen.last_tokens_in == 12
    assert gen.last_tokens_out == 7


@pytest.mark.asyncio
async def test_generator_leaves_tokens_none_when_provider_has_no_usage() -> None:
    """FakeChatModel emits no usage; both tokens fields stay None."""
    llm = FakeChatModel(structured={_Artifact: _Artifact(summary="ok")})
    gen = Generator(
        name="g",
        prompt=_handle(),
        output_schema=_Artifact,
        llm=llm,
        provider="anthropic",
        model_id="fake",
    )
    await gen.invoke(AgentContext(inputs={}))
    assert gen.last_tokens_in is None
    assert gen.last_tokens_out is None


@pytest.mark.asyncio
async def test_generator_renders_prior_verdicts_into_prompt() -> None:
    canned = _Artifact(summary="ok2")
    llm = FakeChatModel(structured={_Artifact: canned})
    gen = Generator(
        name="g",
        prompt=_handle(),
        output_schema=_Artifact,
        llm=llm,
        provider="anthropic",
        model_id="fake",
    )
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
