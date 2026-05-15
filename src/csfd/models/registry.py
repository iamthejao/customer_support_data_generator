"""Build LangChain chat models from AgentLLMConfig."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from csfd.settings import AgentLLMConfig


def build_llm(cfg: AgentLLMConfig) -> BaseChatModel:
    if cfg.provider == "anthropic":
        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens or 4096,
            default_request_timeout=float(cfg.timeout_s),
            anthropic_api_key=SecretStr(cfg.api_key or "test"),
        )
    if cfg.provider == "openai_compat":
        return ChatOpenAI(
            model_name=cfg.model,
            openai_api_base=cfg.base_url,
            openai_api_key=SecretStr(cfg.api_key or "local"),
            temperature=cfg.temperature,
            request_timeout=float(cfg.timeout_s),
        )
    raise ValueError(f"Unknown provider: {cfg.provider}")
