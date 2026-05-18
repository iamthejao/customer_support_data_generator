"""Async SQLite connection manager — mirror of :class:`csfd.storage.db.Database`.

Used inside LangGraph node bodies, which run on the asyncio event loop and
must not call blocking sqlite3 (``blockbuster`` traps it). Sync code paths
(CLI, exporters, migrations runner) continue to use the sync ``Database``.

Both managers point at the same SQLite file. Writers serialize via SQLite's
single-writer model; reads are concurrent under WAL.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite


class AsyncDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        conn = await aiosqlite.connect(self.path, isolation_level=None)
        conn.row_factory = aiosqlite.Row
        try:
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("BEGIN")
            try:
                yield conn
                await conn.execute("COMMIT")
            except Exception:
                await conn.execute("ROLLBACK")
                raise
        finally:
            await conn.close()
