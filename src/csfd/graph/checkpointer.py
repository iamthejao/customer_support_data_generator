"""Async SQLite checkpointer factory for LangGraph persistence."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def async_sqlite_checkpointer(
    db_path: Path | str,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Async context manager yielding an ``AsyncSqliteSaver`` for a SQLite file.

    Used by ``run_phase1`` / ``run_phase2`` for CLI orchestrator runs. The
    LangGraph Studio runtime supplies its own persistence and rejects graphs
    with a custom checkpointer, so the Studio entrypoints don't use this.
    """
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver
