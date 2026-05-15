"""Export a run's artifacts to JSONL (and Parquet — added in P1.19)."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from csfd.storage.db import Database

_TABLES: dict[str, str] = {
    "problems": "SELECT * FROM problems WHERE run_id = ?",
    "kb_articles": "SELECT * FROM kb_articles WHERE run_id = ?",
    "tickets": "SELECT * FROM tickets WHERE run_id = ?",
    "turns": (
        "SELECT turns.* FROM turns "
        "JOIN tickets ON tickets.id = turns.ticket_id "
        "WHERE tickets.run_id = ?"
    ),
    "agent_traces": "SELECT * FROM agent_traces WHERE run_id = ?",
}


def _serialise(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def export_run_to_jsonl(db: Database, run_id: str, *, out_dir: Path) -> list[Path]:
    """Write one .jsonl per artifact table plus a manifest.json. Returns the list of files."""
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    file_checksums: dict[str, str] = {}

    with db.connect() as conn:
        for table, sql in _TABLES.items():
            path = run_dir / f"{table}.jsonl"
            rows = conn.execute(sql, (run_id,)).fetchall()
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    payload = {k: _serialise(r[k]) for k in r.keys()}
                    f.write(json.dumps(payload, separators=(",", ":"), default=str))
                    f.write("\n")
            written.append(path)
            file_checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "files": file_checksums,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return written
