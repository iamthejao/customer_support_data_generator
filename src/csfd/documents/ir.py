"""The document IR (intermediate representation): one typed tree per document.

Every renderer (DOCX, PDF, Markdown, JSON sidecar) walks the same tree, so a
section's id (``sec-6.2``) is the same anchor in every format and in the
retrieval files. Blocks are typed (paragraph, safety message, procedure, list,
table, figure) so renderers can give each its real-document form: a signal-word
warning, numbered steps with expected results, a captioned table.

Ids are shaped so a document can later be shared across runs: ``doc_id`` is
built from the company, the printed document number, the language and the
revision (:func:`make_doc_id`), never from the run id; the database keys rows by
``(run_id, doc_id)``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

DocType = Literal[
    "operating_manual",
    "service_manual",
    "datasheet",
    "parts_list",
    "processing_instructions",
    "work_instruction",
    "kb_article",
    "service_bulletin",
    "release_notes",
    "drawing",
    "maintenance_log",
    "service_report",
    "calibration_certificate",
]
# library: per company / machine model; problem: per Phase 1 problem; case: per case.
Tier = Literal["library", "problem", "case"]
Audience = Literal["customer", "internal"]
DocStatus = Literal["current", "superseded"]
AdmonitionLevel = Literal["danger", "warning", "caution", "notice", "note"]
TableRole = Literal["generic", "parts", "error_codes", "specs", "log"]
MediaType = Literal["image/svg+xml", "image/png", "image/jpeg"]
# Generated from a code template, an SVG written by the model, a CAD projection,
# or an image from the retrieved library.
AssetOrigin = Literal["template", "claude_svg", "cad", "library"]

# Block, step and section ids: lower case, digits, '-' and '.' ("sec-6.2", "proc-6.2-s1").
_ID = re.compile(r"^[a-z][a-z0-9]*(?:[-.][a-z0-9]+)*$")


def _check_id(value: str) -> str:
    if not _ID.fullmatch(value):
        raise ValueError(f"invalid id {value!r}: use lower case, digits, '-' and '.'")
    return value


def make_doc_id(company: str, doc_number: str, language: str, revision: str) -> str:
    """Run-independent document id: ``kalvora-dental:KD-SM-CF600-EN:en:rev-C``."""
    slug = re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-")
    return f"{slug}:{doc_number}:{language}:rev-{revision}"


class Paragraph(BaseModel):
    kind: Literal["paragraph"] = "paragraph"
    id: str
    text: str


class Admonition(BaseModel):
    """A safety message: signal word (ANSI Z535.6 / ISO 3864-2 style) plus text."""

    kind: Literal["admonition"] = "admonition"
    id: str
    level: AdmonitionLevel
    text: str


class Step(BaseModel):
    id: str
    text: str
    expected: str | None = None  # what the reader should see after the step


class Procedure(BaseModel):
    kind: Literal["procedure"] = "procedure"
    id: str
    title: str
    steps: list[Step] = Field(min_length=1)


class ItemList(BaseModel):
    """An unordered list (tools needed, symptoms, related documents)."""

    kind: Literal["list"] = "list"
    id: str
    items: list[str] = Field(min_length=1)


class Table(BaseModel):
    kind: Literal["table"] = "table"
    id: str
    caption: str
    columns: list[str] = Field(min_length=1)
    rows: list[list[str]] = Field(default_factory=list)
    role: TableRole = "generic"

    @model_validator(mode="after")
    def _rectangular(self) -> Table:
        for i, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"table {self.id!r} row {i} has {len(row)} cells, expected {len(self.columns)}"
                )
        return self


class Figure(BaseModel):
    """An image with a self-contained caption and alt text (RAG ingesters read both as text)."""

    kind: Literal["figure"] = "figure"
    id: str
    asset_id: str  # sha256 of the image bytes (see Asset)
    caption: str = Field(min_length=1)
    alt: str = Field(min_length=1)
    description: str = ""  # longer description for the sidecar


Block = Annotated[
    Paragraph | Admonition | Procedure | ItemList | Table | Figure,
    Field(discriminator="kind"),
]


class Section(BaseModel):
    id: str  # stable anchor, e.g. "sec-6.2"
    number: str  # printed number, e.g. "6.2"
    title: str
    blocks: list[Block] = Field(default_factory=list)
    subsections: list[Section] = Field(default_factory=list)
    # Candidate causes this section documents (``csfd.diagnosis.CandidateCause.cause``),
    # so documentation coverage and retrieval links can be traced per cause.
    covers: list[str] = Field(default_factory=list)


class Revision(BaseModel):
    rev: str
    issued: date
    description: str
    author: str


class AppliesTo(BaseModel):
    models: list[str] = Field(default_factory=list)  # seed asset models, e.g. "CF-600"
    serials: list[str] = Field(default_factory=list)  # serial ranges, free text


class Asset(BaseModel):
    """Image bytes a figure points at, with where they came from."""

    asset_id: str
    media_type: MediaType
    content: bytes
    origin: AssetOrigin
    provenance: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        *,
        media_type: MediaType,
        origin: AssetOrigin,
        provenance: dict[str, str] | None = None,
    ) -> Asset:
        return cls(
            asset_id=hashlib.sha256(content).hexdigest(),
            media_type=media_type,
            content=content,
            origin=origin,
            provenance=provenance or {},
        )

    @model_validator(mode="after")
    def _id_is_content_hash(self) -> Asset:
        if self.asset_id != hashlib.sha256(self.content).hexdigest():
            raise ValueError("asset_id must be the sha256 of the content")
        return self


class DocumentIR(BaseModel):
    doc_id: str
    doc_type: DocType
    tier: Tier
    doc_number: str  # the number printed on the document, e.g. "KD-SM-CF600-EN"
    title: str
    subtitle: str | None = None
    company: str
    revision: str
    issue_date: date
    language: str = "en"
    audience: Audience
    status: DocStatus = "current"
    supersedes: str | None = None  # doc_id of the previous revision
    applies_to: AppliesTo = Field(default_factory=AppliesTo)
    revisions: list[Revision] = Field(default_factory=list)
    sections: list[Section] = Field(min_length=1)
    problem_id: str | None = None  # problem tier
    case_uid: str | None = None  # case tier (= lineage.ticket_uid)

    def walk(self) -> Iterator[tuple[Section, int, list[Section]]]:
        """Every section in reading order: (section, depth from 1, its ancestors)."""

        def rec(
            s: Section, depth: int, parents: list[Section]
        ) -> Iterator[tuple[Section, int, list[Section]]]:
            yield s, depth, parents
            for child in s.subsections:
                yield from rec(child, depth + 1, [*parents, s])

        for s in self.sections:
            yield from rec(s, 1, [])

    def section(self, section_id: str) -> Section:
        for s, _, _ in self.walk():
            if s.id == section_id:
                return s
        raise KeyError(f"document {self.doc_id} has no section {section_id!r}")

    def figures(self) -> list[Figure]:
        return [b for s, _, _ in self.walk() for b in s.blocks if isinstance(b, Figure)]

    @property
    def folder_name(self) -> str:
        """File stem used in exports: ``KD-SM-CF600-EN_revC``."""
        stem = f"{self.doc_number}_rev{self.revision}"
        if self.language != "en":
            stem += f"_{self.language}"
        return re.sub(r"[^A-Za-z0-9._-]+", "-", stem)

    @model_validator(mode="after")
    def _check(self) -> DocumentIR:
        seen: set[str] = set()

        def claim(value: str) -> None:
            _check_id(value)
            if value in seen:
                raise ValueError(f"document {self.doc_id}: duplicate id {value!r}")
            seen.add(value)

        for s, _, _ in self.walk():
            claim(s.id)
            for b in s.blocks:
                claim(b.id)
                if isinstance(b, Procedure):
                    for st in b.steps:
                        claim(st.id)
        if self.tier == "problem" and not self.problem_id:
            raise ValueError(f"document {self.doc_id}: a problem-tier document needs problem_id")
        if self.tier == "case" and not self.case_uid:
            raise ValueError(f"document {self.doc_id}: a case-tier document needs case_uid")
        return self
