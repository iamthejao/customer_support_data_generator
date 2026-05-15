"""Pydantic Settings: YAML default + profile overlay + environment variables."""

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
    provider: Literal["anthropic", "openai_compat"]
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout_s: int = 60
    base_url: str | None = None
    api_key: str | None = None


class Phase1Config(BaseModel):
    problem_count: int = 100
    kb_coverage_target_rate: float = 0.7
    dedup_similarity_threshold: float = 0.88
    dedup_method: Literal["lexical", "embedding"] = "lexical"


class TicketsPerProblem(BaseModel):
    has_kb: tuple[int, int] = (3, 5)
    no_kb: tuple[int, int] = (1, 2)


class TicketTypeWeights(BaseModel):
    has_kb: dict[str, float]
    no_kb: dict[str, float]


class Phase2Config(BaseModel):
    tickets_per_problem: TicketsPerProblem = Field(default_factory=TicketsPerProblem)
    ticket_type_weights: TicketTypeWeights
    creative_noise_probability: float = 0.2
    noise_type_weights: dict[str, float] = Field(default_factory=dict)
    voting_policy: Literal["strict_all_pass", "quorum_2_of_3"] = "strict_all_pass"
    retry_exhaustion_policy: Literal["commit_with_warning", "skip"] = "commit_with_warning"
    min_turns_per_ticket: int = 2
    max_turns_per_ticket: int = 12


class ObservabilityConfig(BaseModel):
    structlog_json: bool = True
    langsmith_enabled: bool = False
    langsmith_project: str = "csfd"
    persist_stream_updates: bool = True


class StorageConfig(BaseModel):
    sqlite_path: str = "data/runs.sqlite"
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
    phase1: Phase1Config
    phase2: Phase2Config
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
