import pytest

from csfd.budget.tracker import BudgetTracker
from csfd.errors import BudgetExceededError


def test_tracker_accumulates_tokens_and_cost() -> None:
    t = BudgetTracker(max_tokens=1000, max_usd=1.0)
    t.record(tokens_in=100, tokens_out=50, cost_usd=0.01)
    t.record(tokens_in=200, tokens_out=100, cost_usd=0.02)
    stats = t.stats()
    assert stats["tokens"] == 450
    assert stats["usd"] == pytest.approx(0.03)


def test_tracker_raises_when_tokens_exceeded() -> None:
    t = BudgetTracker(max_tokens=300, max_usd=10.0)
    t.record(tokens_in=200, tokens_out=50, cost_usd=0.01)
    with pytest.raises(BudgetExceededError):
        t.record(tokens_in=100, tokens_out=0, cost_usd=0.01)


def test_tracker_raises_when_usd_exceeded() -> None:
    t = BudgetTracker(max_tokens=10_000, max_usd=0.05)
    t.record(tokens_in=10, tokens_out=10, cost_usd=0.03)
    with pytest.raises(BudgetExceededError):
        t.record(tokens_in=10, tokens_out=10, cost_usd=0.03)
