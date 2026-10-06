"""Parse company_seed.md into a typed CompanyProfile."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_H3 = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")

# The seed section the per-case facts are drawn from (see csfd.facts).
CASE_FACTS_SECTION = "Case facts"


@dataclass(slots=True, frozen=True)
class AssetModel:
    """One entry of the company's catalogue of machines (or products) a customer owns.

    ``serial_format`` uses Faker's ``bothify`` placeholders: ``#`` is a digit and
    ``?`` an upper-case letter.
    """

    model: str
    description: str = ""
    serial_format: str = ""


@dataclass(slots=True, frozen=True)
class SeedPart:
    """A spare part the seed's optional ``### Parts`` table lists for a model.

    An empty ``part_number`` is assigned by the identifier registry
    (:mod:`csfd.documents.registry`).
    """

    model: str
    name: str
    part_number: str = ""


@dataclass(slots=True, frozen=True)
class SeedErrorCode:
    """An error code from the seed's optional ``### Error codes`` table (empty code: assigned)."""

    model: str
    meaning: str
    code: str = ""
    action: str = ""


@dataclass(slots=True, frozen=True)
class SeedDocument:
    """A document number from the seed's optional ``### Documents`` table.

    ``doc_type`` is a document type name (``service_manual``); ``revisions`` run
    oldest to current. Empty values are assigned by the registry.
    """

    model: str
    doc_type: str
    doc_number: str = ""
    revisions: tuple[str, ...] = ()


@dataclass(slots=True)
class CaseFactsCatalogue:
    """Structured values the generator draws each case's facts from.

    ``parts``, ``error_codes`` and ``documents`` come from optional seed tables
    and feed the identifier registry; most seeds leave them out.
    """

    assets: list[AssetModel] = field(default_factory=list)
    caller_roles: list[str] = field(default_factory=list)
    site_locales: list[str] = field(default_factory=list)
    parts: list[SeedPart] = field(default_factory=list)
    error_codes: list[SeedErrorCode] = field(default_factory=list)
    documents: list[SeedDocument] = field(default_factory=list)


@dataclass(slots=True)
class CompanyProfile:
    name: str
    raw_markdown: str
    sections: dict[str, str] = field(default_factory=dict)
    source_path: Path = field(default_factory=lambda: Path("."))
    case_facts: CaseFactsCatalogue = field(default_factory=CaseFactsCatalogue)


def _subsections(text: str) -> dict[str, str]:
    headings = list(_H3.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        out[m.group(1).strip().lower()] = text[m.end() : end].strip()
    return out


def _table_rows(text: str) -> list[dict[str, str]]:
    """Rows of the first Markdown table in ``text``, keyed by lower-cased header."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if not lines:
        return []
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    header = [h.lower() for h in cells[0]]
    return [
        dict(zip(header, row, strict=False))
        for ln, row in zip(lines[1:], cells[1:], strict=True)
        if not _TABLE_RULE.match(ln)
    ]


def _cell(row: dict[str, str], key: str) -> str:
    return row.get(key, "").strip("`* ")


def _parse_case_facts(text: str) -> CaseFactsCatalogue:
    subs = _subsections(text)
    assets = [
        AssetModel(
            model=row["model"].strip("`* "),
            description=row.get("description", ""),
            serial_format=row.get("serial format", "").strip("` "),
        )
        for row in _table_rows(subs.get("assets", ""))
        if row.get("model", "").strip("`* ")
    ]
    return CaseFactsCatalogue(
        assets=assets,
        caller_roles=_BULLET.findall(subs.get("caller roles", "")),
        site_locales=[s.strip("` ") for s in _BULLET.findall(subs.get("site locales", ""))],
        parts=[
            SeedPart(
                model=_cell(r, "model"), name=_cell(r, "name"), part_number=_cell(r, "part number")
            )
            for r in _table_rows(subs.get("parts", ""))
            if _cell(r, "model") and _cell(r, "name")
        ],
        error_codes=[
            SeedErrorCode(
                model=_cell(r, "model"),
                meaning=_cell(r, "meaning"),
                code=_cell(r, "code"),
                action=_cell(r, "action"),
            )
            for r in _table_rows(subs.get("error codes", ""))
            if _cell(r, "model") and _cell(r, "meaning")
        ],
        documents=[
            SeedDocument(
                model=_cell(r, "model"),
                doc_type=_cell(r, "type"),
                doc_number=_cell(r, "number"),
                revisions=tuple(v.strip() for v in _cell(r, "revisions").split(",") if v.strip()),
            )
            for r in _table_rows(subs.get("documents", ""))
            if _cell(r, "model") and _cell(r, "type")
        ],
    )


def parse_company_seed(path: Path) -> CompanyProfile:
    text = path.read_text(encoding="utf-8")
    h1 = _H1.search(text)
    # The H1 may carry a document-title suffix ("Kalvora Dental — Company Profile");
    # the company name is what agents say out loud, so keep only the part before it.
    name = h1.group(1).split(" — ", 1)[0].strip() if h1 else path.stem
    sections: dict[str, str] = {}
    headings = list(_H2.finditer(text))
    for i, m in enumerate(headings):
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections[m.group(1).strip()] = text[start:end].strip()
    return CompanyProfile(
        name=name,
        raw_markdown=text,
        sections=sections,
        source_path=path,
        case_facts=_parse_case_facts(sections.get(CASE_FACTS_SECTION, "")),
    )
