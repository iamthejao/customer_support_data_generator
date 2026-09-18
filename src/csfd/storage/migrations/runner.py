"""Create the CSFD storage schema, idempotently."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from csfd.storage.db import Database

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


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
