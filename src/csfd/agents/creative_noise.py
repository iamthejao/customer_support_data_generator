"""CreativeNoise agent — probabilistically injects realistic complexity into a turn."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from csfd.agents.base import AgentContext, AgentRole
from csfd.agents.generator import Generator
from csfd.prompts.registry import PromptHandle


class TurnModification(BaseModel):
    modified_content: str
    noise_type: str
    rationale: str = Field(default="")


class CreativeNoise(AgentRole):
    """Agent role that returns a modified turn with a noise label."""

    def __init__(
        self,
        *,
        name: str,
        prompt: PromptHandle,
        llm: BaseChatModel,
    ) -> None:
        self.name = name
        self._inner = Generator(name=name, prompt=prompt, output_schema=TurnModification, llm=llm)

    async def invoke(self, ctx: AgentContext) -> TurnModification:
        out: BaseModel = await self._inner.invoke(ctx)
        if not isinstance(out, TurnModification):
            raise TypeError(f"Expected TurnModification, got {type(out).__name__}")
        return out
