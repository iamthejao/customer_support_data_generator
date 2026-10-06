"""Machine-readable sidecars written next to each DOCX/PDF.

* Markdown: the whole document, headings carrying ``{#sec-6.2}`` anchors, safety
  messages as block quotes, tables as pipe tables, figures as images followed by
  their alt text.
* JSON: the IR itself, one **chunk** per section (its own text, heading path,
  PDF page range, Word bookmark, sha256 of the text), and one entry per figure
  (files, caption, alt text, description, provenance).

A chunk is keyed ``<doc_id>#<section_id>``; that id is what retrieval ground
truth points at, so it never depends on how an ingester splits the files.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from csfd.documents.ir import (
    Admonition,
    Asset,
    Block,
    DocumentIR,
    Figure,
    ItemList,
    Paragraph,
    Procedure,
    Section,
    Table,
    bookmark_name,
)
from csfd.documents.render_docx import SIGNALS

SCHEMA = "csfd.document/1"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    section_id: str
    number: str
    title: str
    heading_path: list[str]  # "6 Calibration", "6.2 Silver test"
    depth: int
    text: str  # the section's own blocks (subsections are their own chunks)
    covers: list[str]
    pdf_pages: tuple[int, int] | None  # first and last PDF page, when a PDF was rendered
    docx_bookmark: str
    sha256: str


def chunk_id(doc_id: str, section_id: str) -> str:
    return f"{doc_id}#{section_id}"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def block_markdown(block: Block, figure_paths: Mapping[str, str]) -> list[str]:
    """One block as Markdown lines."""
    if isinstance(block, Paragraph):
        return [block.text]
    if isinstance(block, Admonition):
        word, _ = SIGNALS[block.level]
        return [f"> **{word}:** {block.text}"]
    if isinstance(block, Procedure):
        lines = [f"**Procedure: {block.title}**", ""]
        for n, step in enumerate(block.steps, 1):
            result = f" *Result: {step.expected}*" if step.expected else ""
            lines.append(f"{n}. {step.text}{result}")
        return lines
    if isinstance(block, ItemList):
        return [f"- {item}" for item in block.items]
    if isinstance(block, Table):
        lines = [
            f"*{block.caption}*",
            "",
            "| " + " | ".join(_cell(c) for c in block.columns) + " |",
            "|" + "---|" * len(block.columns),
        ]
        lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in block.rows]
        return lines
    assert isinstance(block, Figure)
    path = figure_paths.get(block.id, "")
    lines = [f"![{block.caption}]({path})", "", f"*{block.caption}.* {block.alt}"]
    if block.description:
        lines.append(block.description)
    return lines


def section_text(section: Section, figure_paths: Mapping[str, str]) -> str:
    """The section's own blocks as Markdown, blank-line separated."""
    parts = ["\n".join(block_markdown(b, figure_paths)) for b in section.blocks]
    return "\n\n".join(parts)


def render_markdown(doc: DocumentIR, figure_paths: Mapping[str, str]) -> str:
    lines = [f"# {doc.title}", ""]
    if doc.subtitle:
        lines += [f"*{doc.subtitle}*", ""]
    lines += [
        "| Field | Value |",
        "|---|---|",
        f"| Document number | {_cell(doc.doc_number)} |",
        f"| Revision | {_cell(doc.revision)} |",
        f"| Issue date | {doc.issue_date.isoformat()} |",
        f"| Status | {doc.status} |",
        f"| Applies to | {_cell(', '.join(doc.applies_to.models + doc.applies_to.serials) or '-')} |",
        f"| Audience | {doc.audience} |",
        "",
    ]
    if doc.revisions:
        lines += [
            "## Revision history",
            "",
            "| Rev. | Date | Description | Author |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {_cell(r.rev)} | {r.issued.isoformat()} | {_cell(r.description)} | {_cell(r.author)} |"
            for r in doc.revisions
        ]
        lines.append("")
    for section, depth, _ in doc.walk():
        lines += [
            f"{'#' * min(depth + 1, 6)} {section.number} {section.title} {{#{section.id}}}",
            "",
        ]
        text = section_text(section, figure_paths)
        if text:
            lines += [text, ""]
    return "\n".join(lines)


def chunks(
    doc: DocumentIR,
    figure_paths: Mapping[str, str],
    pdf_pages: Mapping[str, tuple[int, int]] | None = None,
) -> list[Chunk]:
    out: list[Chunk] = []
    for section, depth, parents in doc.walk():
        text = section_text(section, figure_paths)
        out.append(
            Chunk(
                chunk_id=chunk_id(doc.doc_id, section.id),
                doc_id=doc.doc_id,
                section_id=section.id,
                number=section.number,
                title=section.title,
                heading_path=[f"{s.number} {s.title}" for s in [*parents, section]],
                depth=depth,
                text=text,
                covers=list(section.covers),
                pdf_pages=pdf_pages.get(section.id) if pdf_pages is not None else None,
                docx_bookmark=bookmark_name(section.id),
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return out


def sidecar(
    doc: DocumentIR,
    doc_chunks: list[Chunk],
    assets: Mapping[str, Asset],
    figure_files: Mapping[str, list[str]],
    files: Mapping[str, str],
) -> dict[str, Any]:
    """The JSON sidecar: IR, chunks, figures, and the sha256 of every rendered file."""
    figures = []
    for section, _, _ in doc.walk():
        for block in section.blocks:
            if isinstance(block, Figure):
                asset = assets[block.asset_id]
                figures.append(
                    {
                        "figure_id": block.id,
                        "section_id": section.id,
                        "files": figure_files.get(block.id, []),
                        "caption": block.caption,
                        "alt": block.alt,
                        "description": block.description,
                        "asset_id": asset.asset_id,
                        "media_type": asset.media_type,
                        "origin": asset.origin,
                        "provenance": asset.provenance,
                    }
                )
    return {
        "schema": SCHEMA,
        "document": doc.model_dump(mode="json"),
        "chunks": [c.model_dump(mode="json") for c in doc_chunks],
        "figures": figures,
        "files": dict(files),
    }
