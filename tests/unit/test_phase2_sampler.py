from __future__ import annotations

from csfd.phases.phase1_kb.state import CommittedProblem, ProblemDraft
from csfd.phases.phase2_cases.sampler import (
    sample_ticket_count,
    sample_ticket_type,
)
from csfd.ticket_types.definitions import TicketType
from csfd.utils.rng import derive_rng


def _problem(has_kb: bool) -> CommittedProblem:
    return CommittedProblem(
        id="p",
        draft=ProblemDraft(title="t", description="d", category="c", severity="low"),
        has_kb=has_kb,
        coverage_reasoning="",
        coverage_confidence="low",
    )


def test_sample_ticket_type_no_kb_forces_l3() -> None:
    rng = derive_rng(1, "x")
    weights_has_kb = {"docs_request": 0.10, "l1": 0.55, "l2": 0.25, "l3": 0.10}
    weights_no_kb = {"l3": 1.0}
    for _ in range(100):
        t = sample_ticket_type(
            _problem(has_kb=False),
            rng,
            weights_has_kb=weights_has_kb,
            weights_no_kb=weights_no_kb,
        )
        assert t == TicketType.L3


def test_sample_ticket_type_has_kb_uses_weighted_distribution() -> None:
    rng = derive_rng(42, "dist-check")
    weights_has_kb = {"docs_request": 0.10, "l1": 0.55, "l2": 0.25, "l3": 0.10}
    weights_no_kb = {"l3": 1.0}
    seen: dict[str, int] = {"docs_request": 0, "l1": 0, "l2": 0, "l3": 0}
    for _ in range(1000):
        t = sample_ticket_type(
            _problem(has_kb=True),
            rng,
            weights_has_kb=weights_has_kb,
            weights_no_kb=weights_no_kb,
        )
        seen[t.value] += 1
    assert seen["l1"] > seen["l2"] > seen["docs_request"]
    assert seen["l1"] > 400
    assert all(v > 0 for v in seen.values())


def test_sample_ticket_type_deterministic_under_same_seed() -> None:
    weights_has_kb = {"l1": 1.0}
    weights_no_kb = {"l3": 1.0}
    p = _problem(has_kb=True)
    rng_a = derive_rng(7, "x")
    rng_b = derive_rng(7, "x")
    seq_a = [
        sample_ticket_type(p, rng_a, weights_has_kb=weights_has_kb, weights_no_kb=weights_no_kb)
        for _ in range(50)
    ]
    seq_b = [
        sample_ticket_type(p, rng_b, weights_has_kb=weights_has_kb, weights_no_kb=weights_no_kb)
        for _ in range(50)
    ]
    assert seq_a == seq_b


def test_sample_ticket_count_in_range() -> None:
    rng = derive_rng(1, "count")
    for _ in range(50):
        n = sample_ticket_count(rng, range_inclusive=(3, 5))
        assert 3 <= n <= 5


def test_sample_ticket_count_single_value_range() -> None:
    rng = derive_rng(1, "count")
    assert sample_ticket_count(rng, range_inclusive=(2, 2)) == 2
