"""TicketState + draft/persona/stats Pydantic models for the Phase 2 subgraph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from csfd.agents.base import Verdict
from csfd.phases.phase1_kb.state import CommittedArticle, CommittedProblem
from csfd.seeds.company import CompanyProfile


class CustomerPersona(BaseModel):
    """Customer-side persona attached to a ticket."""

    name: str
    tier: str
    tone: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentPersona(BaseModel):
    """Support-rep persona attached to a ticket."""

    name: str
    tier: str
    expertise: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TicketDraft(BaseModel):
    """An LLM- or sampler-produced candidate ticket before persistence."""

    problem_id: str
    kb_article_id: str | None
    ticket_type: Literal["docs_request", "l1", "l2", "l3"]
    priority: Literal["low", "medium", "high", "critical"]
    subject: str
    customer_persona: CustomerPersona
    agent_persona: AgentPersona
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnDraft(BaseModel):
    """An LLM-generated candidate turn before persistence."""

    speaker: Literal["customer", "agent", "system"]
    content: str
    intent: (
        Literal[
            "question",
            "clarification",
            "resolution",
            "escalation",
            "thanks",
            "closing",
        ]
        | None
    ) = None
    speaker_persona: str | None = None
    kb_references: list[str] = Field(default_factory=list)
    noise_applied: bool = False
    noise_type: str | None = None


class Phase2Stats(BaseModel):
    tickets_committed: int = 0
    turns_committed: int = 0
    turns_with_noise: int = 0
    total_tokens: int = 0
    total_usd: float = 0.0


class CommittedTicket(BaseModel):
    """A ticket committed during the run."""

    id: str
    draft: TicketDraft
    status: Literal["resolved", "unresolved", "escalated"] = "resolved"
    ground_truth: dict[str, Any] | None = None
    quality_flag: str | None = None
    unresolved_issues: list[Verdict] = Field(default_factory=list)


class CommittedTurn(BaseModel):
    """A turn committed during the run."""

    id: str
    ticket_id: str
    turn_index: int
    draft: TurnDraft
    quality_flag: str | None = None
    unresolved_issues: list[Verdict] = Field(default_factory=list)


class TicketState(BaseModel):
    """In-memory working state for the Phase 2 LangGraph subgraph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    parent_run_id: str
    run_seed: int
    company: CompanyProfile

    problem_pool: list[CommittedProblem] = Field(default_factory=list)
    kb_by_problem: dict[str, CommittedArticle] = Field(default_factory=dict)

    current_ticket: CommittedTicket | None = None
    turns_committed: list[CommittedTurn] = Field(default_factory=list)
    current_turn_draft: TurnDraft | None = None

    noise_applied: bool = False
    noise_type: str | None = None
    turn_index: int = 0

    verdicts: Annotated[list[Verdict], operator.add] = Field(default_factory=list)
    retry_attempt: int = 0

    stats: Phase2Stats = Field(default_factory=Phase2Stats)
