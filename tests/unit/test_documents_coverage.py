"""Documentation coverage: exact, seeded, one value per distinct cause text."""

from __future__ import annotations

from collections import Counter

import pytest

from csfd.diagnosis import CandidateCause, DiagnosisPlan
from csfd.documents.coverage import Coverage, answer_source, plan_coverage
from csfd.settings import DocumentsConfig

PROPORTIONS: dict[Coverage, float] = {"documented": 0.6, "partial": 0.2, "agent_only": 0.2}


def _plan(*causes: str) -> DiagnosisPlan:
    return DiagnosisPlan(candidate_causes=[CandidateCause(cause=c) for c in causes])


def _plans(n: int) -> list[tuple[str, DiagnosisPlan]]:
    return [(f"r:p:{i:04d}", _plan(f"cause {i}a", f"cause {i}b")) for i in range(n)]


def test_counts_are_exact_over_the_distinct_causes() -> None:
    coverage = plan_coverage(_plans(5), PROPORTIONS, seed=1)  # 10 distinct causes
    counts = Counter(level for causes in coverage.values() for level in causes.values())
    assert counts == {"documented": 6, "partial": 2, "agent_only": 2}


def test_the_same_cause_text_gets_the_same_value_everywhere() -> None:
    plans = [
        ("r:p:0000", _plan("Loose PSU connector", "Failing fan")),
        ("r:p:0001", _plan("loose PSU-connector.", "Dust")),
        ("r:p:0002", _plan("Worn belt", "Bad sensor")),
    ]
    for seed in range(20):
        coverage = plan_coverage(plans, PROPORTIONS, seed=seed)
        assert (
            coverage["r:p:0000"]["Loose PSU connector"]
            == (coverage["r:p:0001"]["loose PSU-connector."])
        )
        # Five distinct causes: 3 documented, 1 partial, 1 agent only.
        distinct = {
            "loose": coverage["r:p:0000"]["Loose PSU connector"],
            "fan": coverage["r:p:0000"]["Failing fan"],
            "dust": coverage["r:p:0001"]["Dust"],
            "belt": coverage["r:p:0002"]["Worn belt"],
            "sensor": coverage["r:p:0002"]["Bad sensor"],
        }
        assert Counter(distinct.values()) == {"documented": 3, "partial": 1, "agent_only": 1}


def test_coverage_is_seeded() -> None:
    plans = _plans(6)
    assert plan_coverage(plans, PROPORTIONS, seed=3) == plan_coverage(plans, PROPORTIONS, seed=3)
    assert any(
        plan_coverage(plans, PROPORTIONS, seed=3) != plan_coverage(plans, PROPORTIONS, seed=s)
        for s in range(4, 10)
    )


def test_zero_weights_and_no_causes() -> None:
    coverage = plan_coverage(_plans(2), {"documented": 0, "agent_only": 1}, seed=1)
    assert {lv for c in coverage.values() for lv in c.values()} == {"agent_only"}
    assert plan_coverage([], PROPORTIONS, seed=1) == {}


def test_answer_source_follows_the_root_cause_coverage() -> None:
    assert answer_source("documented") == "documents"
    assert answer_source("partial") == "partial"
    assert answer_source("agent_only") == "agent_knowledge"


def test_config_rejects_unusable_proportions() -> None:
    assert DocumentsConfig().coverage_proportions == PROPORTIONS
    with pytest.raises(ValueError, match="positive sum"):
        DocumentsConfig.model_validate({"coverage_proportions": {"documented": 0}})
    with pytest.raises(ValueError, match="non-negative"):
        DocumentsConfig.model_validate({"coverage_proportions": {"documented": 1, "partial": -1}})
    with pytest.raises(ValueError):
        DocumentsConfig.model_validate({"coverage_proportions": {"maybe": 1}})
