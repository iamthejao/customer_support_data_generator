from pathlib import Path

import pytest

from csfd.storage.db import Database


def test_database_creates_file_and_pragmas(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    with db.connect() as conn:
        cur = conn.execute("PRAGMA foreign_keys")
        assert cur.fetchone()[0] == 1
        cur = conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0].lower() == "wal"
    assert tmp_db_path.exists()


def test_database_context_commits_on_success(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    with db.connect() as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t (v) VALUES (?)", ("hello",))
    with db.connect() as conn:
        row = conn.execute("SELECT v FROM t").fetchone()
        assert row[0] == "hello"


def test_database_context_rolls_back_on_exception(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    with db.connect() as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")

    with pytest.raises(RuntimeError):
        with db.connect() as conn:
            conn.execute("INSERT INTO t (id) VALUES (1)")
            raise RuntimeError("boom")

    with db.connect() as conn:
        rows: int = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert rows == 0
