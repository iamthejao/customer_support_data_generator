"""Pydantic Settings: YAML default + profile overlay + environment variables.

This module hosts the proportion-based schema (``problem_database``, ``tickets``,
``validation``) used by the deterministic LangGraph pipeline. All sections
required by the pipeline are non-Optional so misconfiguration fails at load
time rather than at runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
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
    provider: Literal["anthropic", "openai_compat", "claude_code_cli", "codex_cli"]
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


class DialogueConfig(BaseModel):
    turn_cap: int = 20


Channel = Literal["email", "phone"]
Disfluency = Literal["none", "light", "moderate"]


class PhoneConfig(BaseModel):
    """Spoken-style controls, used only when ``tickets.channel == "phone"``."""

    disfluency: Disfluency = "light"


class CalendarConfig(BaseModel):
    """Deterministic wall-clock placement of contacts.

    Start times are derived from ``start`` plus seeded offsets, never from the
    current clock, so the same config and seed place every contact at the same
    timestamp. Contacts fall on weekdays within ``business_hours`` (local hours
    in ``start``'s timezone).
    """

    start: datetime = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
    span_days: int = Field(default=20, ge=1)
    business_hours: tuple[int, int] = (8, 18)

    @field_validator("business_hours")
    @classmethod
    def _check_business_hours(cls, v: tuple[int, int]) -> tuple[int, int]:
        open_h, close_h = v
        if not 0 <= open_h < close_h <= 24:
            raise ValueError("business_hours must satisfy 0 <= open < close <= 24")
        return v


CallbackReason = Literal["follow_up", "dropped"]


def _default_callback_reasons() -> dict[CallbackReason, float]:
    return {"follow_up": 0.7, "dropped": 0.3}


class RoundsConfig(BaseModel):
    """How many contacts (calls / emails) one case takes before it closes.

    Every case is one allocation slot. A case with N rounds yields N related
    contacts that share a ``case_uid``: rounds 1..N-1 end without closing the
    case, for one of ``callback_reasons``, and round N concludes it.
    """

    # Contacts per case -> proportion of cases, applied with largest-remainder
    # rounding over tickets.total. The default keeps one contact per case.
    proportions: dict[int, float] = Field(default_factory=lambda: {1: 1.0})
    # Why a non-final contact ends: an agreed next step that needs time
    # (follow_up), or the call cutting off mid-conversation (dropped).
    callback_reasons: dict[CallbackReason, float] = Field(default_factory=_default_callback_reasons)
    # Hours between consecutive contact starts, drawn log-uniformly, then moved
    # into business hours. The minimum keeps consecutive calls from overlapping.
    gap_hours: tuple[float, float] = (2.0, 72.0)

    @field_validator("proportions")
    @classmethod
    def _check_proportions(cls, v: dict[int, float]) -> dict[int, float]:
        if not v or any(k < 1 for k in v):
            raise ValueError("rounds.proportions keys are contact counts and must be >= 1")
        if sum(v.values()) <= 0:
            raise ValueError("rounds.proportions must have a positive sum")
        return v

    @field_validator("callback_reasons")
    @classmethod
    def _check_reasons(cls, v: dict[CallbackReason, float]) -> dict[CallbackReason, float]:
        if not v or sum(v.values()) <= 0:
            raise ValueError("rounds.callback_reasons must have a positive sum")
        return v

    @field_validator("gap_hours")
    @classmethod
    def _check_gap(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not 1.0 <= lo <= hi:
            raise ValueError("rounds.gap_hours must satisfy 1 <= min <= max")
        return v


class TicketsConfig(BaseModel):
    total: int = 100
    type_proportions: dict[str, float]
    assignment_strategy: Literal["uniform", "complexity_weighted"] = "complexity_weighted"
    dialogue: DialogueConfig = Field(default_factory=DialogueConfig)
    tier_proportions: dict[str, float]
    tone_proportions_per_type: dict[str, dict[str, float]]
    channel: Channel = "email"
    phone: PhoneConfig = Field(default_factory=PhoneConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    rounds: RoundsConfig = Field(default_factory=RoundsConfig)


class ValidationConfig(BaseModel):
    enabled: bool = True
    max_retries: int = 2


class EmbeddingConfig(BaseModel):
    enabled: bool = False
    provider: Literal["openai_compat"] = "openai_compat"
    base_url: str = "http://localhost:11434/v1"
    model: str = "embeddinggemma:300m"
    dim: int | None = 768
    threshold: float = 0.85
    text_template: Literal[
        "title_summary",
        "title_summary_background",
        "title_summary_symptoms_root_cause",
    ] = "title_summary"
    timeout_s: int = 30


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
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
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
