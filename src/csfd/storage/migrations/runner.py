"""Apply numbered .sql migrations idempotently."""

from __future__ import annotations

import re
from pathlib import Path

from csfd.storage.db import Database

_MIGRATIONS_DIR = Path(__file__).parent
_MIGRATION_PATTERN = re.compile(r"^(\d{3})_.*\.sql$")


def _discover() -> list[tuple[str, Path]]:
    files = []
    for p in sorted(_MIGRATIONS_DIR.iterdir()):
        m = _MIGRATION_PATTERN.match(p.name)
        if m:
            files.append((m.group(1), p))
    return files


def applied_versions(db: Database) -> list[str]:
    with db.connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  version TEXT PRIMARY KEY,"
            "  applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    return [r[0] for r in rows]


def apply_migrations(db: Database) -> list[str]:
    """Apply pending migrations; return list of newly-applied versions."""
    import sqlite3

    already = set(applied_versions(db))
    newly_applied: list[str] = []
    for version, path in _discover():
        if version in already:
            continue
        sql = path.read_text(encoding="utf-8")
        # executescript() auto-commits, so bypass transaction wrapper
        conn = sqlite3.connect(db.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
        conn.commit()
        conn.close()
        newly_applied.append(version)
    return newly_applied
