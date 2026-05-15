"""AgentFactory composes prompt registry + LLM builder + per-agent config into agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from csfd.agents.checker import Checker
from csfd.agents.creative_noise import CreativeNoise
from csfd.agents.generator import Generator
from csfd.prompts.registry import PromptRegistry
from csfd.settings import AgentLLMConfig

LLMBuilder = Callable[[AgentLLMConfig], BaseChatModel]


@dataclass(slots=True)
class AgentFactory:
    prompts: PromptRegistry
    llm_builder: LLMBuilder
    agent_configs: dict[str, AgentLLMConfig]

    def _llm_for(self, name: str) -> BaseChatModel:
        if name not in self.agent_configs:
            raise KeyError(f"No agent config for: {name}")
        return self.llm_builder(self.agent_configs[name])

    def build_generator(
        self,
        *,
        name: str,
        prompt_name: str,
        output_schema_factory: Callable[[], type[BaseModel]],
    ) -> Generator:
        return Generator(
            name=name,
            prompt=self.prompts.get(prompt_name),
            output_schema=output_schema_factory(),
            llm=self._llm_for(name),
        )

    def build_checker(self, *, name: str, prompt_name: str) -> Checker:
        return Checker(
            name=name,
            prompt=self.prompts.get(prompt_name),
            llm=self._llm_for(name),
        )

    def build_creative_noise(self, *, name: str, prompt_name: str) -> CreativeNoise:
        return CreativeNoise(
            name=name,
            prompt=self.prompts.get(prompt_name),
            llm=self._llm_for(name),
        )
