"""Generator agent — produces a structured artifact from a prompt + context."""

from __future__ import annotations

import time

import structlog
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from csfd.agents.base import AgentContext, AgentRole
from csfd.prompts.registry import PromptHandle

_log = structlog.get_logger(__name__)


class Generator(AgentRole):
    """Agent role that emits an artifact of a given Pydantic schema."""

    def __init__(
        self,
        *,
        name: str,
        prompt: PromptHandle,
        output_schema: type[BaseModel],
        llm: BaseChatModel,
        provider: str,
        model_id: str,
    ) -> None:
        self.name = name
        self.prompt = prompt
        self.output_schema = output_schema
        self.llm = llm
        self.provider = provider
        self.model_id = model_id
        self.last_rendered_prompt: str | None = None
        self.last_tokens_in: int | None = None
        self.last_tokens_out: int | None = None

    async def invoke(self, ctx: AgentContext) -> BaseModel:
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        rendered = self.prompt.template.render(
            inputs=ctx.inputs,
            prior_committed=ctx.prior_committed,
            retry_attempt=ctx.retry_attempt,
            prior_verdicts=[v.model_dump() for v in ctx.prior_verdicts],
        )
        self.last_rendered_prompt = rendered
        runnable = self.llm.with_structured_output(self.output_schema)
        usage_cb = UsageMetadataCallbackHandler()
        _log.info(
            "agent.invoke.start",
            agent=self.name,
            schema=self.output_schema.__name__,
            prompt_chars=len(rendered),
            retry_attempt=ctx.retry_attempt,
        )
        t0 = time.perf_counter()
        try:
            result = await runnable.ainvoke(rendered, config={"callbacks": [usage_cb]})
        except Exception as e:
            _log.error(
                "agent.invoke.error",
                agent=self.name,
                schema=self.output_schema.__name__,
                duration_s=round(time.perf_counter() - t0, 2),
                error=repr(e),
            )
            self.last_tokens_in = None
            self.last_tokens_out = None
            raise
        # Sum tokens across any model keys the callback observed (single-key
        # in practice). Providers that do not emit usage_metadata (FakeChatModel,
        # the local claude CLI wrapper) leave the dict empty → fields stay None.
        tin = sum(v.get("input_tokens", 0) for v in usage_cb.usage_metadata.values()) or None
        tout = sum(v.get("output_tokens", 0) for v in usage_cb.usage_metadata.values()) or None
        self.last_tokens_in = tin
        self.last_tokens_out = tout
        duration_s = round(time.perf_counter() - t0, 2)
        if not isinstance(result, BaseModel):
            _log.error(
                "agent.invoke.bad_type",
                agent=self.name,
                got=type(result).__name__,
                duration_s=duration_s,
            )
            raise TypeError(f"Expected {self.output_schema.__name__}, got {type(result).__name__}")
        _log.info(
            "agent.invoke.end",
            agent=self.name,
            schema=self.output_schema.__name__,
            duration_s=duration_s,
        )
        return result
