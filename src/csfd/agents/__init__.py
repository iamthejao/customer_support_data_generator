"""csfd.agents — agent role framework."""

from csfd.agents.base import AgentContext, AgentRole, Issue, Verdict
from csfd.agents.checker import Checker
from csfd.agents.creative_noise import CreativeNoise, TurnModification
from csfd.agents.factory import AgentFactory
from csfd.agents.generator import Generator
from csfd.agents.tracing import record_agent_trace

__all__ = [
    "AgentContext",
    "AgentFactory",
    "AgentRole",
    "Checker",
    "CreativeNoise",
    "Generator",
    "Issue",
    "TurnModification",
    "Verdict",
    "record_agent_trace",
]
