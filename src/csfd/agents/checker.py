"""Checker agent — emits a Verdict on a candidate artifact."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from csfd.agents.base import AgentContext, AgentRole, Verdict
from csfd.agents.generator import Generator
from csfd.prompts.registry import PromptHandle


class Checker(AgentRole):
    """Agent role that always emits a Verdict.

    Internally delegates to a Generator bound to the Verdict schema, then forces
    the verdict's `checker` field to match this checker's own name (the LLM may
    fill it but we own the identity).
    """

    def __init__(
        self,
        *,
        name: str,
        prompt: PromptHandle,
        llm: BaseChatModel,
    ) -> None:
        self.name = name
        self._inner = Generator(name=name, prompt=prompt, output_schema=Verdict, llm=llm)

    async def invoke(self, ctx: AgentContext) -> Verdict:
        out: BaseModel = await self._inner.invoke(ctx)
        if not isinstance(out, Verdict):
            raise TypeError(f"Checker expected Verdict, got {type(out).__name__}")
        return out.model_copy(update={"checker": self.name})
