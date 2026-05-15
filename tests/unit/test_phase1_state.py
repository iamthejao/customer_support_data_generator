from uuid import uuid4

from csfd.agents.base import Verdict  # noqa: F401 - ensures reducer wiring
from csfd.phases.phase1_kb.state import (
    CoverageDecision,
    KBArticleDraft,
    KBState,
    PhaseStats,
    ProblemDraft,
)
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue


def _empty_company() -> CompanyProfile:
    return CompanyProfile(name="Acme", raw_markdown="# Acme")


def _empty_scenarios() -> ScenarioCatalogue:
    return ScenarioCatalogue()


def test_kb_state_defaults() -> None:
    state = KBState(
        run_id=str(uuid4()),
        run_seed=42,
        company=_empty_company(),
        scenarios=_empty_scenarios(),
    )
    assert state.problems_committed == []
    assert state.current_problem_draft is None
    assert state.current_article_draft is None
    assert state.verdicts == []
    assert state.retry_attempt == 0
    assert isinstance(state.stats, PhaseStats)


def test_kb_state_verdict_reducer_appends() -> None:
    """Verify the Annotated reducer for verdicts is operator.add (list concat)."""
    from typing import get_args, get_type_hints

    hints = get_type_hints(KBState, include_extras=True)
    annotated = hints["verdicts"]
    args = get_args(annotated)
    import operator

    assert args[1] is operator.add


def test_problem_draft_required_fields() -> None:
    p = ProblemDraft(
        title="Login fails",
        description="Customer cannot log in",
        category="Authentication",
        severity="medium",
    )
    assert p.title == "Login fails"


def test_kb_article_draft_required_fields() -> None:
    a = KBArticleDraft(
        title="How to reset",
        content_markdown="## Step 1\n...",
        troubleshooting_steps=[{"step": "do X", "expected_result": "Y"}],
    )
    assert a.title == "How to reset"
    assert len(a.troubleshooting_steps) == 1


def test_coverage_decision_required_fields() -> None:
    d = CoverageDecision(has_kb=True, reasoning="Common issue", confidence="high")
    assert d.has_kb is True
