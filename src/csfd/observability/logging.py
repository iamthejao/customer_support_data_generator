"""structlog configuration with contextvars-based correlation."""
from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

import structlog


def configure_logging(*, json_output: bool = True, stream: TextIO | None = None) -> None:
    """Configure structlog + stdlib logging.

    Args:
        json_output: if True, emit JSON; else key=value.
        stream: where to write log lines (defaults to sys.stderr).
    """
    target: TextIO = stream if stream is not None else sys.stderr

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(file=target),
        cache_logger_on_first_use=False,
    )
    # mirror stdlib root logger to the same target for libraries that use logging
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(target)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def bind_context(**kwargs: Any) -> None:
    """Bind key-value pairs into the contextvars-scoped log context for this task."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all contextvars bound to the current task's log context."""
    structlog.contextvars.clear_contextvars()
