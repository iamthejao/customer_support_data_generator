"""AgentFactory composes prompt registry + LLM builder + per-agent config into agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from csfd.agents.checker import Checker
from csfd.agents.generator import Generator
from csfd.prompts.registry import PromptRegistry
from csfd.settings import AgentLLMConfig

LLMBuilder = Callable[[AgentLLMConfig], BaseChatModel]

_ROLE_FALLBACK: dict[str, str] = {
    "problem_brainstorm": "generator",
    "resolution_generator": "generator",
    "combined_problem_check": "combined_checker",
    "combined_resolution_check": "combined_checker",
}


@dataclass(slots=True)
class AgentFactory:
    prompts: PromptRegistry
    llm_builder: LLMBuilder
    agent_configs: dict[str, AgentLLMConfig]

    def _cfg_for(self, name: str) -> AgentLLMConfig:
        if name in self.agent_configs:
            return self.agent_configs[name]
        role = _ROLE_FALLBACK.get(name)
        if role is not None and role in self.agent_configs:
            return self.agent_configs[role]
        raise KeyError(f"No agent config for: {name}")

    def _llm_for(self, name: str) -> BaseChatModel:
        return self.llm_builder(self._cfg_for(name))

    def build_generator(
        self,
        *,
        name: str,
        prompt_name: str,
        output_schema_factory: Callable[[], type[BaseModel]],
    ) -> Generator:
        cfg = self._cfg_for(name)
        return Generator(
            name=name,
            prompt=self.prompts.get(prompt_name),
            output_schema=output_schema_factory(),
            llm=self.llm_builder(cfg),
            provider=cfg.provider,
            model_id=cfg.model,
        )

    def build_checker(self, *, name: str, prompt_name: str) -> Checker:
        cfg = self._cfg_for(name)
        return Checker(
            name=name,
            prompt=self.prompts.get(prompt_name),
            llm=self.llm_builder(cfg),
            provider=cfg.provider,
            model_id=cfg.model,
        )
