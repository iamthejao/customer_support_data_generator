"""Data access for a run's identifier registry (see :mod:`csfd.documents.registry`).

One row per machine model, kept in catalogue order (insertion order).
"""

from __future__ import annotations

from collections.abc import Mapping

from csfd.documents.registry import ProductFacts
from csfd.storage.db import Database
from csfd.storage.db_async import AsyncDatabase

_SELECT_SQL = "SELECT facts_json FROM product_facts WHERE run_id = ? ORDER BY rowid"


def _by_model(rows: list[str]) -> dict[str, ProductFacts]:
    facts = [ProductFacts.model_validate_json(r) for r in rows]
    return {f.model: f for f in facts}


class ProductFactsRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def areplace(
        self, adb: AsyncDatabase, run_id: str, registry: Mapping[str, ProductFacts]
    ) -> None:
        """Store the run's registry, removing what it had before."""
        async with adb.connect() as conn:
            await conn.execute("DELETE FROM product_facts WHERE run_id = ?", (run_id,))
            await conn.executemany(
                "INSERT INTO product_facts (run_id, model, source, facts_json) VALUES (?, ?, ?, ?)",
                [(run_id, f.model, f.source, f.model_dump_json()) for f in registry.values()],
            )

    async def aupdate(self, adb: AsyncDatabase, run_id: str, facts: ProductFacts) -> None:
        """Rewrite one model's entry in place (keeps its position)."""
        async with adb.connect() as conn:
            await conn.execute(
                "UPDATE product_facts SET source = ?, facts_json = ? WHERE run_id = ? AND model = ?",
                (facts.source, facts.model_dump_json(), run_id, facts.model),
            )

    async def alist_for_run(self, adb: AsyncDatabase, run_id: str) -> dict[str, ProductFacts]:
        async with adb.connect() as conn:
            cur = await conn.execute(_SELECT_SQL, (run_id,))
            rows = await cur.fetchall()
        return _by_model([r["facts_json"] for r in rows])

    def list_for_run(self, run_id: str) -> dict[str, ProductFacts]:
        with self.db.connect() as conn:
            rows = conn.execute(_SELECT_SQL, (run_id,)).fetchall()
        return _by_model([r["facts_json"] for r in rows])
