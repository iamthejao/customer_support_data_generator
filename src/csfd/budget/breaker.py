"""Circuit breaker around LLM transport. Trips on repeated TransportError."""
from __future__ import annotations

from pybreaker import CircuitBreaker


def build_breaker(*, fail_max: int = 5, reset_timeout: int = 60) -> CircuitBreaker:
    """Return a CircuitBreaker. Wrap only LLM transport calls with this."""
    return CircuitBreaker(fail_max=fail_max, reset_timeout=reset_timeout)
