"""Shared SqliteSaver factory for LangGraph checkpoint persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def build_sqlite_checkpointer(db_path: Path | str) -> SqliteSaver:
    """Open a SqliteSaver wrapping a shared SQLite connection.

    The connection uses ``check_same_thread=False`` so async LangGraph
    invocations may reuse it across event-loop tasks.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)
