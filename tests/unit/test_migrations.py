from pathlib import Path

from csfd.storage.db import Database
from csfd.storage.migrations.runner import applied_versions, apply_migrations


def test_apply_migrations_creates_all_tables(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    with db.connect() as conn:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    for required in {
        "runs",
        "problems",
        "incoming_requests",
        "resolutions",
        "lineage",
        "agent_traces",
        "schema_migrations",
    }:
        assert required in names


def test_apply_migrations_is_idempotent(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    apply_migrations(db)
    assert set(applied_versions(db)) == {"001", "002", "003"}


def test_problems_table_columns(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    with db.connect() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(problems)").fetchall()}
    assert {
        "id",
        "run_id",
        "title",
        "summary",
        "background",
        "category",
        "complexity",
        "resolution_hints_json",
        "quality_flag",
        "created_at",
    }.issubset(cols)
