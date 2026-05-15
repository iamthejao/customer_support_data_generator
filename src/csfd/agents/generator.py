"""Generator agent — produces a structured artifact from a prompt + context."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from csfd.agents.base import AgentContext, AgentRole
from csfd.prompts.registry import PromptHandle


class Generator(AgentRole):
    """Agent role that emits an artifact of a given Pydantic schema."""

    def __init__(
        self,
        *,
        name: str,
        prompt: PromptHandle,
        output_schema: type[BaseModel],
        llm: BaseChatModel,
    ) -> None:
        self.name = name
        self.prompt = prompt
        self.output_schema = output_schema
        self.llm = llm
        self.last_rendered_prompt: str | None = None

    async def invoke(self, ctx: AgentContext) -> BaseModel:
        rendered = self.prompt.template.render(
            inputs=ctx.inputs,
            prior_committed=ctx.prior_committed,
            retry_attempt=ctx.retry_attempt,
            prior_verdicts=[v.model_dump() for v in ctx.prior_verdicts],
        )
        self.last_rendered_prompt = rendered
        runnable = self.llm.with_structured_output(self.output_schema)
        result = await runnable.ainvoke(rendered)
        if not isinstance(result, BaseModel):
            raise TypeError(f"Expected {self.output_schema.__name__}, got {type(result).__name__}")
        return result
