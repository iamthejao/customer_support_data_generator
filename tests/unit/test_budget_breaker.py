import pytest
from pybreaker import CircuitBreakerError

from csfd.budget.breaker import build_breaker
from csfd.errors import TransportError


def test_breaker_trips_after_threshold() -> None:
    br = build_breaker(fail_max=3, reset_timeout=60)

    @br
    def always_fail() -> None:
        raise TransportError("boom")

    # First 2 failures should raise TransportError
    for _ in range(2):
        with pytest.raises(TransportError):
            always_fail()

    # 3rd call reaches fail_max threshold and trips the circuit
    with pytest.raises(CircuitBreakerError):
        always_fail()

    # 4th call also raises CircuitBreakerError (circuit is open)
    with pytest.raises(CircuitBreakerError):
        always_fail()


def test_breaker_resets_on_success() -> None:
    br = build_breaker(fail_max=2, reset_timeout=60)

    state = {"fail": True}

    @br
    def maybe_fail() -> str:
        if state["fail"]:
            raise TransportError("nope")
        return "ok"

    with pytest.raises(TransportError):
        maybe_fail()
    state["fail"] = False
    assert maybe_fail() == "ok"
