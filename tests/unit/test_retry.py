import pytest

from csfd.errors import SchemaValidationError, TransportError
from csfd.utils.retry import with_transport_retry


@pytest.mark.asyncio
async def test_with_transport_retry_succeeds_after_transient_failures() -> None:
    calls = {"n": 0}

    @with_transport_retry(max_attempts=4, initial_seconds=0.0, max_seconds=0.0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransportError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_with_transport_retry_gives_up_after_max_attempts() -> None:
    calls = {"n": 0}

    @with_transport_retry(max_attempts=3, initial_seconds=0.0, max_seconds=0.0)
    async def always_fail() -> None:
        calls["n"] += 1
        raise TransportError("dead")

    with pytest.raises(TransportError):
        await always_fail()
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_with_transport_retry_does_not_retry_other_errors() -> None:
    calls = {"n": 0}

    @with_transport_retry(max_attempts=3, initial_seconds=0.0, max_seconds=0.0)
    async def wrong_error() -> None:
        calls["n"] += 1
        raise SchemaValidationError(errors=[], raw_output="")

    with pytest.raises(SchemaValidationError):
        await wrong_error()
    assert calls["n"] == 1
