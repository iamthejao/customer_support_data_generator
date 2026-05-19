"""Export a run's artifacts to JSONL (and Parquet — added in P1.19)."""

from __future__ import annotations

import hashlib
import json
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from csfd.storage.db import Database

_TABLES: dict[str, str] = {
    "problems": "SELECT * FROM problems WHERE run_id = ?",
    "incoming_requests": "SELECT * FROM incoming_requests WHERE run_id = ?",
    "resolutions": "SELECT * FROM resolutions WHERE run_id = ?",
    "lineage": "SELECT * FROM lineage WHERE run_id = ?",
    "agent_traces": "SELECT * FROM agent_traces WHERE run_id = ?",
    "problem_embeddings": "SELECT * FROM problem_embeddings WHERE run_id = ?",
}


def _serialise(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _row_to_payload(table: str, row: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {k: _serialise(row[k]) for k in row.keys()}
    if table == "problem_embeddings":
        blob = row["vector"]
        dim = int(row["dim"])
        payload["vector"] = list(struct.unpack(f"<{dim}f", blob))
    return payload


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
                    payload = _row_to_payload(table, r)
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


def export_run_to_parquet(db: Database, run_id: str, *, out_dir: Path) -> list[Path]:
    """Write one .parquet per artifact table. Returns the list of files."""
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with db.connect() as conn:
        for table, sql in _TABLES.items():
            rows = conn.execute(sql, (run_id,)).fetchall()
            if not rows:
                continue
            cols = list(rows[0].keys())
            data: dict[str, list[Any]] = {c: [] for c in cols}
            for r in rows:
                payload = _row_to_payload(table, r)
                for c in cols:
                    data[c].append(payload[c])
            path = run_dir / f"{table}.parquet"
            pq.write_table(pa.Table.from_pydict(data), path)
            written.append(path)
    return written
