"""KBState + draft Pydantic models for the Phase 1 KB-generation subgraph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from csfd.agents.base import Verdict
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue


class ProblemDraft(BaseModel):
    title: str
    description: str
    category: str
    severity: Literal["low", "medium", "high", "critical"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class KBArticleDraft(BaseModel):
    title: str
    content_markdown: str
    troubleshooting_steps: list[dict[str, str]] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoverageDecision(BaseModel):
    has_kb: bool
    reasoning: str
    confidence: Literal["low", "medium", "high"]


class PhaseStats(BaseModel):
    problems_generated: int = 0
    problems_committed: int = 0
    articles_committed: int = 0
    total_tokens: int = 0
    total_usd: float = 0.0


class CommittedProblem(BaseModel):
    id: str
    draft: ProblemDraft
    has_kb: bool
    coverage_reasoning: str
    coverage_confidence: str
    quality_flag: str | None = None
    unresolved_issues: list[Verdict] = Field(default_factory=list)


class CommittedArticle(BaseModel):
    id: str
    problem_id: str
    draft: KBArticleDraft
    quality_flag: str | None = None
    unresolved_issues: list[Verdict] = Field(default_factory=list)


class KBState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    run_seed: int
    company: CompanyProfile
    scenarios: ScenarioCatalogue

    problems_committed: list[CommittedProblem] = Field(default_factory=list)
    articles_committed: list[CommittedArticle] = Field(default_factory=list)

    current_problem_draft: ProblemDraft | None = None
    current_article_draft: KBArticleDraft | None = None

    verdicts: Annotated[list[Verdict], operator.add] = Field(default_factory=list)
    retry_attempt: int = 0

    stats: PhaseStats = Field(default_factory=PhaseStats)
