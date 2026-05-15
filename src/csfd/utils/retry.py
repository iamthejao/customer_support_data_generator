"""Tenacity-based retry wrappers for transport-layer errors."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from csfd.errors import TransportError

T = TypeVar("T")


def with_transport_retry(
    *,
    max_attempts: int = 5,
    initial_seconds: float = 1.0,
    max_seconds: float = 30.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator: retry an async function on TransportError with exp-backoff + jitter."""

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_jitter(initial=initial_seconds, max=max_seconds),
                retry=retry_if_exception_type(TransportError),
                reraise=True,
            ):
                with attempt:
                    return await fn(*args, **kwargs)
            raise RuntimeError("unreachable")

        return wrapper

    return decorator
