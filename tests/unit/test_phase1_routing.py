from uuid import uuid4

from langgraph.types import Send

from csfd.agents.base import Issue, Verdict
from csfd.phases.phase1_kb.routing import (
    aggregate_verdicts,
    dispatch_problem_checkers,
    route_problem_verdict,
)
from csfd.phases.phase1_kb.state import KBState, ProblemDraft
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue


def _state(*, verdicts: list[Verdict] | None = None, retry: int = 0) -> KBState:
    return KBState(
        run_id=str(uuid4()),
        run_seed=1,
        company=CompanyProfile(name="A", raw_markdown="# A"),
        scenarios=ScenarioCatalogue(),
        current_problem_draft=ProblemDraft(
            title="t",
            description="d",
            category="c",
            severity="low",
        ),
        verdicts=verdicts or [],
        retry_attempt=retry,
    )


def test_dispatch_problem_checkers_sends_to_three_nodes() -> None:
    state = _state()
    sends = dispatch_problem_checkers(state)
    targets = sorted(s.node for s in sends)
    assert targets == [
        "problem_background_check",
        "problem_consistency_check",
        "problem_scenario_check",
    ]
    for s in sends:
        assert isinstance(s, Send)


def test_route_problem_verdict_pass_returns_commit() -> None:
    state = _state(
        verdicts=[
            Verdict(checker="consistency", passed=True),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ]
    )
    assert route_problem_verdict(state, max_retries=3) == "commit"


def test_route_problem_verdict_fail_under_budget_returns_regenerate() -> None:
    state = _state(
        verdicts=[
            Verdict(
                checker="consistency",
                passed=False,
                issues=[
                    Issue(severity="error", location="x", rule_violated="r", explanation="e"),
                ],
            ),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ],
        retry=1,
    )
    assert route_problem_verdict(state, max_retries=3) == "regenerate"


def test_route_problem_verdict_fail_over_budget_returns_commit_with_warning() -> None:
    state = _state(
        verdicts=[
            Verdict(
                checker="consistency",
                passed=False,
                issues=[
                    Issue(severity="error", location="x", rule_violated="r", explanation="e"),
                ],
            ),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ],
        retry=3,
    )
    assert route_problem_verdict(state, max_retries=3) == "commit_with_warning"


def test_aggregate_verdicts_is_noop_passthrough() -> None:
    state = _state(verdicts=[Verdict(checker="x", passed=True)])
    update = aggregate_verdicts(state)
    assert update == {}
