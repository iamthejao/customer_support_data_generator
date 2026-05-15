"""Routing helpers for the Phase 1 LangGraph subgraph."""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from csfd.phases.phase1_kb.state import KBState


def dispatch_problem_checkers(state: KBState) -> list[Send]:
    """Fan out the candidate problem to the 3 checkers."""
    payload = {"current_problem_draft": state.current_problem_draft}
    return [
        Send("problem_consistency_check", payload),
        Send("problem_background_check", payload),
        Send("problem_scenario_check", payload),
    ]


def aggregate_verdicts(state: KBState) -> dict[str, Any]:
    """No-op fan-in barrier; the Annotated reducer already accumulated verdicts.

    Provides a stable seam for adding logic later (e.g. quorum voting).
    """
    return {}


def route_problem_verdict(state: KBState, *, max_retries: int) -> str:
    """Decide next step after the 3 checkers' verdicts are aggregated."""
    if all(v.passed for v in state.verdicts):
        return "commit"
    if state.retry_attempt >= max_retries:
        return "commit_with_warning"
    return "regenerate"
