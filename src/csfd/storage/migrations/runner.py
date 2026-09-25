"""Create the CSFD storage schema, idempotently."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from csfd.storage.db import Database

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class StaleSchemaError(RuntimeError):
    """The database was created by an older schema; it is not upgraded in place."""


def _columns(conn: sqlite3.Connection) -> dict[str, set[str]]:
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {
        t: {row[1] for row in conn.execute(f"PRAGMA table_info({t})")}
        for (t,) in tables
        if not t.startswith("sqlite_")
    }


def check_schema(db: Database) -> None:
    """Raise :class:`StaleSchemaError` if a schema table in ``db`` lacks a column."""
    expected_conn = sqlite3.connect(":memory:")
    try:
        expected_conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        expected = _columns(expected_conn)
    finally:
        expected_conn.close()
    conn = sqlite3.connect(db.path)
    try:
        actual = _columns(conn)
    finally:
        conn.close()
    missing = [
        f"{table}.{col}"
        for table, cols in expected.items()
        if table in actual
        for col in sorted(cols - actual[table])
    ]
    if missing:
        raise StaleSchemaError(
            f"{db.path} was created by an older csfd schema (missing {', '.join(missing)}) "
            "and is not upgraded in place. Delete or move it, then run `csfd db-migrate`."
        )


def apply_migrations(db: Database) -> None:
    """Create every table/index in the schema if it does not already exist."""
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    # executescript() auto-commits, so bypass the transaction wrapper.
    conn = sqlite3.connect(db.path, check_same_thread=False)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()
    check_schema(db)
