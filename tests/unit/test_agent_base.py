import pytest
from csfd.agents.base import AgentContext, Issue, Verdict
from pydantic import ValidationError


def test_issue_requires_severity_and_location() -> None:
    issue = Issue(
        severity="error",
        location="turn[2].assistant_message",
        rule_violated="L1_must_not_mention_internal_tools",
        explanation="Mentioned Jira ticket id",
        suggested_fix="Remove or replace",
    )
    assert issue.severity == "error"
    assert issue.suggested_fix == "Remove or replace"


def test_issue_severity_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        Issue(**{"severity": "critical", "location": "x", "rule_violated": "r", "explanation": "e"})


def test_verdict_pass_with_no_issues() -> None:
    v = Verdict(checker="consistency", passed=True)
    assert v.passed is True
    assert v.issues == []


def test_verdict_fail_carries_issues() -> None:
    v = Verdict(
        checker="background",
        passed=False,
        issues=[
            Issue(
                severity="warning",
                location="problem.description",
                rule_violated="tone_company_voice",
                explanation="Too informal",
            )
        ],
    )
    assert v.passed is False
    assert len(v.issues) == 1


def test_agent_context_defaults() -> None:
    ctx = AgentContext(inputs={"a": 1})
    assert ctx.inputs == {"a": 1}
    assert ctx.prior_committed == {}
    assert ctx.retry_attempt == 0
    assert ctx.prior_verdicts == []
