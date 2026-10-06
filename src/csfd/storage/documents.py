"""Data access for a run's supporting documents (see :mod:`csfd.documents`).

A build replaces everything a run had before (documents, images, links,
answer sources) in one transaction, so re-running ``csfd documents <run_id>``
is idempotent and never mixes two builds.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from csfd.documents.build import AnswerSource, BuildResult, DocumentLink
from csfd.documents.ir import Asset, DocumentIR
from csfd.storage.db import Database

_TABLES = ("document_links", "document_cases", "document_assets", "documents")


class DocumentRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def replace_for_run(self, run_id: str, result: BuildResult, made_by: dict[str, str]) -> None:
        """Store a build, removing whatever the run had before."""
        now = datetime.now(UTC).isoformat()
        assets = {a.asset_id: a for built in result.documents for a in built.assets}
        with self.db.connect() as conn:
            for table in _TABLES:
                conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            for built in result.documents:
                doc = built.document
                ir_json = doc.model_dump_json()
                conn.execute(
                    """
                    INSERT INTO documents
                      (run_id, doc_id, tier, doc_type, doc_number, revision, status,
                       audience, language, problem_id, case_uid, builder, ir_json,
                       ir_sha256, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        doc.doc_id,
                        doc.tier,
                        doc.doc_type,
                        doc.doc_number,
                        doc.revision,
                        doc.status,
                        doc.audience,
                        doc.language,
                        doc.problem_id,
                        doc.case_uid,
                        made_by[doc.doc_id],
                        ir_json,
                        hashlib.sha256(ir_json.encode("utf-8")).hexdigest(),
                        now,
                    ),
                )
            for asset in assets.values():
                conn.execute(
                    """
                    INSERT INTO document_assets
                      (run_id, asset_id, media_type, origin, provenance_json, content)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        asset.asset_id,
                        asset.media_type,
                        asset.origin,
                        json.dumps(asset.provenance, ensure_ascii=False, sort_keys=True),
                        asset.content,
                    ),
                )
            for link in result.links:
                conn.execute(
                    """
                    INSERT INTO document_links
                      (run_id, case_uid, round_index, doc_id, section_id, relation, grade, basis)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        link.case_uid,
                        link.round_index,
                        link.doc_id,
                        link.section_id,
                        link.relation,
                        link.grade,
                        link.basis,
                    ),
                )
            for case in result.cases:
                conn.execute(
                    "INSERT INTO document_cases (run_id, case_uid, answer_source) VALUES (?, ?, ?)",
                    (run_id, case.case_uid, case.answer_source),
                )

    def has_documents(self, run_id: str) -> bool:
        """Whether the run has documents; False for a database created before documents existed."""
        with self.db.connect() as conn:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
            ).fetchone()
            if table is None:
                return False
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row["n"]) > 0

    def list_for_run(self, run_id: str) -> list[DocumentIR]:
        """The run's documents: library, then problem, then case tier; by number and revision."""
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT ir_json FROM documents WHERE run_id = ?
                ORDER BY CASE tier WHEN 'library' THEN 0 WHEN 'problem' THEN 1 ELSE 2 END,
                         doc_number, revision, doc_id
                """,
                (run_id,),
            ).fetchall()
        return [DocumentIR.model_validate_json(r["ir_json"]) for r in rows]

    def assets_for_run(self, run_id: str) -> dict[str, Asset]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM document_assets WHERE run_id = ? ORDER BY asset_id", (run_id,)
            ).fetchall()
        return {
            r["asset_id"]: Asset(
                asset_id=r["asset_id"],
                media_type=r["media_type"],
                content=bytes(r["content"]),
                origin=r["origin"],
                provenance=json.loads(r["provenance_json"]),
            )
            for r in rows
        }

    def links_for_run(self, run_id: str) -> list[DocumentLink]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM document_links WHERE run_id = ?
                ORDER BY case_uid, round_index, doc_id, section_id, relation
                """,
                (run_id,),
            ).fetchall()
        return [
            DocumentLink(
                case_uid=r["case_uid"],
                round_index=r["round_index"],
                doc_id=r["doc_id"],
                section_id=r["section_id"],
                relation=r["relation"],
                grade=r["grade"],
                basis=r["basis"],
            )
            for r in rows
        ]

    def answer_sources(self, run_id: str) -> dict[str, AnswerSource]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT case_uid, answer_source FROM document_cases WHERE run_id = ?", (run_id,)
            ).fetchall()
        return {r["case_uid"]: r["answer_source"] for r in rows}
