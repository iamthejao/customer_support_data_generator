"""Ticket type enum and per-type metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TicketType(StrEnum):
    DOCS_REQUEST = "docs_request"
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"


class ProblemComplexity(StrEnum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass(slots=True, frozen=True)
class TicketTypeMetadata:
    avg_turns: int
    persona_label: str
    requires_kb: bool
    description: str
    complexity_preference: tuple[ProblemComplexity, ...] = field(default_factory=tuple)


TICKET_TYPE_METADATA: dict[TicketType, TicketTypeMetadata] = {
    TicketType.DOCS_REQUEST: TicketTypeMetadata(
        avg_turns=2,
        persona_label="Documentation Rep",
        requires_kb=True,
        description="Customer asks for documentation; agent provides links + summary.",
        complexity_preference=(ProblemComplexity.SIMPLE, ProblemComplexity.MEDIUM),
    ),
    TicketType.L1: TicketTypeMetadata(
        avg_turns=3,
        persona_label="L1 Support",
        requires_kb=True,
        description="First-line rep; quick resolution; high reliance on KB.",
        complexity_preference=(ProblemComplexity.SIMPLE, ProblemComplexity.MEDIUM),
    ),
    TicketType.L2: TicketTypeMetadata(
        avg_turns=5,
        persona_label="L2 Support",
        requires_kb=True,
        description="Escalated; structured troubleshooting; KB + judgment.",
        complexity_preference=(ProblemComplexity.MEDIUM, ProblemComplexity.COMPLEX),
    ),
    TicketType.L3: TicketTypeMetadata(
        avg_turns=7,
        persona_label="L3 Domain Specialist",
        requires_kb=False,
        description="Domain specialist; novel/edge-case resolution; no KB available.",
        complexity_preference=(ProblemComplexity.COMPLEX, ProblemComplexity.MEDIUM),
    ),
}
