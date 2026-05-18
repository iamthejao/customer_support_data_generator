from pathlib import Path

import pytest

from csfd.storage.db_async import AsyncDatabase


@pytest.mark.asyncio
async def test_async_database_round_trip(tmp_path: Path) -> None:
    db = AsyncDatabase(tmp_path / "runs.sqlite")
    async with db.connect() as conn:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO t (name) VALUES (?)", ("alice",))
    async with db.connect() as conn:
        cur = await conn.execute("SELECT name FROM t")
        rows = await cur.fetchall()
    assert [r["name"] for r in rows] == ["alice"]


@pytest.mark.asyncio
async def test_async_database_rolls_back_on_exception(tmp_path: Path) -> None:
    db = AsyncDatabase(tmp_path / "runs.sqlite")
    async with db.connect() as conn:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
    with pytest.raises(RuntimeError):
        async with db.connect() as conn:
            await conn.execute("INSERT INTO t (name) VALUES ('bob')")
            raise RuntimeError("boom")
    async with db.connect() as conn:
        cur = await conn.execute("SELECT COUNT(*) AS n FROM t")
        row = await cur.fetchone()
    assert row is not None
    assert row["n"] == 0
