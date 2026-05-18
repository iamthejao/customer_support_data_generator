"""Pydantic Settings: YAML default + profile overlay + environment variables.

This module hosts the proportion-based schema (``problem_database``, ``tickets``,
``validation``) used by the deterministic LangGraph pipeline. All sections
required by the pipeline are non-Optional so misconfiguration fails at load
time rather than at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BudgetConfig(BaseModel):
    max_tokens_per_run: int = 5_000_000
    max_usd_per_run: float = 10.0
    max_retries_per_artifact: int = 3


class PipelineConfig(BaseModel):
    version: str = "0.1.0"
    run_seed: int | None = None
    budget: BudgetConfig = Field(default_factory=BudgetConfig)


class AgentLLMConfig(BaseModel):
    provider: Literal["anthropic", "openai_compat", "claude_code_cli"]
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout_s: int = 60
    base_url: str | None = None
    api_key: str | None = None


# ---- Proportion-based schema (used by the deterministic pipeline) ----


class ProblemDatabaseConfig(BaseModel):
    count: int = 10
    complexity_proportions: dict[str, float] = Field(
        default_factory=lambda: {"simple": 0.3, "medium": 0.4, "complex": 0.3}
    )


class TicketsConfig(BaseModel):
    total: int = 100
    type_proportions: dict[str, float]
    assignment_strategy: Literal["uniform", "complexity_weighted"] = "complexity_weighted"
    turns_per_type: dict[str, int]
    tier_proportions: dict[str, float]
    tone_proportions_per_type: dict[str, dict[str, float]]


class ValidationConfig(BaseModel):
    enabled: bool = True
    max_retries: int = 2


# ---- Common / cross-cutting ----


class ObservabilityConfig(BaseModel):
    structlog_json: bool = True
    langsmith_enabled: bool = False
    langsmith_project: str = "csfd"
    persist_stream_updates: bool = True


class StorageConfig(BaseModel):
    sqlite_path: str = "data/runs.sqlite"
    checkpoint_sqlite_path: str = ".langgraph_api/checkpoints.sqlite"
    exports_dir: str = "data/exports"
    exports_format: list[str] = Field(default_factory=lambda: ["jsonl", "parquet"])


class EnvSecrets(BaseSettings):
    """Environment-sourced secrets and overrides."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    anthropic_api_key: str | None = None
    local_base_url: str | None = None
    local_api_key: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None


class AppSettings(BaseModel):
    pipeline: PipelineConfig
    agents: dict[str, AgentLLMConfig]
    problem_database: ProblemDatabaseConfig
    tickets: TicketsConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    observability: ObservabilityConfig
    storage: StorageConfig
    env: EnvSecrets = Field(default_factory=EnvSecrets)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(
    *, default_path: str = "config/default.yaml", profile: str | None = None
) -> AppSettings:
    with open(default_path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if profile:
        profile_path = Path("config/profiles") / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        with open(profile_path, encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        data = _deep_merge(data, overlay)

    return AppSettings(**data)
