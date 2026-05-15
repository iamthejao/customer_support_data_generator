"""Base agent types: Issue, Verdict, AgentContext, AgentRole."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class Issue(BaseModel):
    """Structured defect raised by a checker."""

    severity: Literal["error", "warning"]
    location: str
    rule_violated: str
    explanation: str
    suggested_fix: str | None = None


class Verdict(BaseModel):
    """A checker's judgement on a candidate artifact."""

    checker: str
    passed: bool
    issues: list[Issue] = Field(default_factory=list)


class AgentContext(BaseModel):
    """Inputs passed to every agent invocation."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    prior_committed: dict[str, Any] = Field(default_factory=dict)
    retry_attempt: int = 0
    prior_verdicts: list[Verdict] = Field(default_factory=list)


class AgentRole(ABC):
    """ABC for all agent roles. Subclasses implement `invoke`."""

    name: str

    @abstractmethod
    async def invoke(self, ctx: AgentContext) -> BaseModel:
        """Run the agent against the context and return its typed output."""
