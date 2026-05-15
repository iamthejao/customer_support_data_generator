"""Typed exception hierarchy for the csfd pipeline."""

from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base class for all csfd pipeline errors."""


class TransportError(PipelineError):
    """Network-level failure (5xx, timeout, connection reset). Retried at the call layer."""


class SchemaValidationError(PipelineError):
    """LLM output failed Pydantic validation. Routed back to the generator with feedback."""

    def __init__(self, *, errors: list[dict[str, Any]], raw_output: str) -> None:
        self.errors = errors
        self.raw_output = raw_output
        super().__init__(f"Schema validation failed with {len(errors)} error(s)")


class BudgetExceededError(PipelineError):
    """Token or USD budget hit. The run aborts; committed artifacts persist."""

    def __init__(self, *, stats: dict[str, Any]) -> None:
        self.stats = stats
        super().__init__(f"Budget exceeded: {stats}")


class ConvergenceError(PipelineError):
    """Verdict-issue list identical across two consecutive retries — generator is stuck."""
