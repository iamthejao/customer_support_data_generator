"""Export a run's documents: DOCX, PDF, sidecars, figures and retrieval files.

Layout under ``<out>/<run_id>/documents/``::

    index.jsonl                      one line per document (ids, type, tier, files + sha256)
    docs/<doc_number>_rev<rev>/
        <stem>.docx, <stem>.pdf      as configured in documents.formats
        <stem>.md, <stem>.json       Markdown and JSON sidecars (always)
        assets/<figure_id>.<ext>     figure images (SVG figures also as PNG)
    rag/                             BEIR-style retrieval files (see csfd.documents.rag)
    manifest.json                    sha256 of every file above

A run without documents exports nothing. Output depends only on the stored
documents, so exporting twice gives identical files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from csfd.documents.figures import extension, svg_to_png
from csfd.documents.ir import Asset, DocumentIR
from csfd.documents.rag import Query, write_rag_files
from csfd.documents.render_docx import render_docx
from csfd.documents.render_pdf import render_pdf
from csfd.documents.sidecar import Chunk, chunks, render_markdown, sidecar
from csfd.settings import DocumentFormat
from csfd.storage.db import Database
from csfd.storage.documents import DocumentRepo

_QUERIES_SQL = """
SELECT l.ticket_uid, l.problem_id, l.ticket_type, ir.subject, ir.body
FROM lineage l
JOIN incoming_requests ir ON ir.id = l.incoming_request_id
WHERE l.run_id = ?
ORDER BY l.slot_index
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _queries(db: Database, run_id: str) -> list[Query]:
    with db.connect() as conn:
        rows = conn.execute(_QUERIES_SQL, (run_id,)).fetchall()
    return [
        Query(
            case_uid=r["ticket_uid"],
            problem_id=r["problem_id"],
            ticket_type=r["ticket_type"],
            subject=r["subject"],
            text=r["body"],
        )
        for r in rows
    ]


def _export_document(
    doc: DocumentIR,
    assets: dict[str, Asset],
    folder: Path,
    formats: Sequence[DocumentFormat],
) -> tuple[list[Path], list[Chunk], dict[str, str]]:
    written: list[Path] = []
    files: dict[str, str] = {}
    figure_paths: dict[str, str] = {}
    figure_files: dict[str, list[str]] = {}
    for fig in doc.figures():
        asset = assets[fig.asset_id]
        rel = f"assets/{fig.id}.{extension(asset)}"
        written.append(_write(folder / rel, asset.content))
        figure_files[fig.id] = [rel]
        figure_paths[fig.id] = rel
        if asset.media_type == "image/svg+xml":
            png_rel = f"assets/{fig.id}.png"
            written.append(_write(folder / png_rel, svg_to_png(asset.content)))
            figure_files[fig.id].append(png_rel)
            figure_paths[fig.id] = png_rel
    stem = doc.folder_name
    pages = None
    if "docx" in formats:
        written.append(_write(folder / f"{stem}.docx", render_docx(doc, assets)))
    if "pdf" in formats:
        pdf, pages = render_pdf(doc, assets)
        written.append(_write(folder / f"{stem}.pdf", pdf))
    written.append(
        _write(folder / f"{stem}.md", render_markdown(doc, figure_paths).encode("utf-8"))
    )
    for path in written:
        files[path.relative_to(folder).as_posix()] = _sha(path.read_bytes())
    doc_chunks = chunks(doc, figure_paths, pages)
    payload = sidecar(doc, doc_chunks, assets, figure_files, files)
    side = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    written.append(_write(folder / f"{stem}.json", side))
    files[f"{stem}.json"] = _sha(side)
    return written, doc_chunks, files


def export_run_documents(
    db: Database,
    run_id: str,
    *,
    out_dir: Path,
    formats: Sequence[DocumentFormat] = ("docx", "pdf"),
) -> list[Path]:
    """Write ``<out_dir>/<run_id>/documents/``; returns the files written ([] without documents)."""
    repo = DocumentRepo(db)
    if not repo.has_documents(run_id):
        return []
    docs = repo.list_for_run(run_id)
    assets = repo.assets_for_run(run_id)
    root = out_dir / run_id / "documents"
    # The folder is entirely generated: clear it so no file of an earlier export lingers.
    if root.exists():
        shutil.rmtree(root)
    written: list[Path] = []
    index_rows = []
    exported: list[tuple[DocumentIR, list[Chunk]]] = []
    for doc in docs:
        folder = root / "docs" / doc.folder_name
        paths, doc_chunks, files = _export_document(doc, assets, folder, formats)
        written += paths
        exported.append((doc, doc_chunks))
        index_rows.append(
            {
                "doc_id": doc.doc_id,
                "doc_number": doc.doc_number,
                "revision": doc.revision,
                "status": doc.status,
                "supersedes": doc.supersedes,
                "tier": doc.tier,
                "doc_type": doc.doc_type,
                "audience": doc.audience,
                "language": doc.language,
                "title": doc.title,
                "applies_to": doc.applies_to.model_dump(),
                "problem_id": doc.problem_id,
                "case_uid": doc.case_uid,
                "folder": f"docs/{doc.folder_name}",
                "files": files,
            }
        )
    index = root / "index.jsonl"
    index.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in index_rows), encoding="utf-8"
    )
    written.append(index)
    written += write_rag_files(
        root,
        exported,
        _queries(db, run_id),
        repo.links_for_run(run_id),
        repo.answer_sources(run_id),
    )
    manifest = {
        "run_id": run_id,
        "files": {p.relative_to(root).as_posix(): _sha(p.read_bytes()) for p in sorted(written)},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written
