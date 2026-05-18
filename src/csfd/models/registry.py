"""Build LangChain chat models from AgentLLMConfig."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from csfd.models.claude_code_cli import ClaudeCodeCLIModel
from csfd.settings import AgentLLMConfig


def build_llm(cfg: AgentLLMConfig) -> BaseChatModel:
    if cfg.provider == "claude_code_cli":
        return ClaudeCodeCLIModel(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.timeout_s,
        )
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
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=SecretStr(cfg.api_key or "local"),
            temperature=cfg.temperature,
            timeout=float(cfg.timeout_s),
        )
    raise ValueError(f"Unknown provider: {cfg.provider}")
