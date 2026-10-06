"""Render a document IR to PDF with Typst (the ``typst`` package bundles the compiler).

One template (``templates/document.typ``) serves every document type; the IR
goes in as JSON. The PDF has a running header, "Page X of Y", an outline
(bookmarks) built from the headings, a tagged structure, and optionally the
PDF/UA-1 (accessibility) or PDF/A-2b (archiving) profile. Only Typst's built-in
fonts are used and the creation date is pinned to the issue date, so the bytes
are the same on every machine.

Besides the PDF, rendering returns the page range of each section, read back
from the compiled document, for the sidecar's ``pdf_pages``.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime, time
from importlib.resources import files
from pathlib import Path

import typst

from csfd.documents.figures import extension
from csfd.documents.ir import Asset, DocumentIR

_TEMPLATE = "document.typ"


def template_source() -> str:
    """The Typst template shipped inside the package."""
    return files("csfd.documents").joinpath("templates", _TEMPLATE).read_text(encoding="utf-8")


def render_pdf(
    doc: DocumentIR, assets: Mapping[str, Asset]
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    """The document as tagged PDF/UA-1 bytes, and each section's (first, last) page."""
    with tempfile.TemporaryDirectory(prefix="csfd-doc-") as tmp:
        root = Path(tmp)
        (root / _TEMPLATE).write_text(template_source(), encoding="utf-8")
        figure_files: dict[str, str] = {}
        for fig in doc.figures():
            asset = assets[fig.asset_id]
            name = f"{fig.id}.{extension(asset)}"
            (root / name).write_bytes(asset.content)
            figure_files[fig.id] = name
        payload = json.dumps({"doc": doc.model_dump(mode="json"), "files": figure_files})
        compiler = typst.Compiler(
            str(root / _TEMPLATE),
            root=str(root),
            ignore_system_fonts=True,
            sys_inputs={"payload": payload},
        )
        pdf = compiler.compile(
            format="pdf",
            timestamp=datetime.combine(doc.issue_date, time(0, 0), tzinfo=UTC),
            pdf_standards=["ua-1"],
        )
        marks = json.loads(compiler.query("<secmark>", field="value"))
    assert isinstance(pdf, bytes)
    return pdf, _page_ranges(marks)


def _page_ranges(marks: list[dict[str, object]]) -> dict[str, tuple[int, int]]:
    start: dict[str, int] = {}
    end: dict[str, int] = {}
    for m in marks:
        sid, page = str(m["id"]), int(str(m["page"]))
        (start if m["edge"] == "start" else end)[sid] = page
    return {sid: (page, end.get(sid, page)) for sid, page in start.items()}
