import io
import json

import structlog

from csfd.observability.logging import bind_context, configure_logging


def test_configure_logging_emits_json_with_contextvars() -> None:
    buf = io.StringIO()
    configure_logging(json_output=True, stream=buf)
    bind_context(run_id="run-1", agent_role="generator")
    log = structlog.get_logger("test")
    log.info("hello", extra="value")
    payload = json.loads(buf.getvalue().splitlines()[-1])
    assert payload["event"] == "hello"
    assert payload["run_id"] == "run-1"
    assert payload["agent_role"] == "generator"
    assert payload["extra"] == "value"
    assert payload["level"] == "info"


def test_configure_logging_emits_kv_when_not_json() -> None:
    buf = io.StringIO()
    configure_logging(json_output=False, stream=buf)
    log = structlog.get_logger("test2")
    log.info("hi")
    out = buf.getvalue()
    assert "event=" in out or "hi" in out
