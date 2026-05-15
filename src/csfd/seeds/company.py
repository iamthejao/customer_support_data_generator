"""Parse company_seed.md into a typed CompanyProfile."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(slots=True)
class CompanyProfile:
    name: str
    raw_markdown: str
    sections: dict[str, str] = field(default_factory=dict)
    source_path: Path = field(default_factory=lambda: Path("."))


def parse_company_seed(path: Path) -> CompanyProfile:
    text = path.read_text(encoding="utf-8")
    h1 = _H1.search(text)
    name = h1.group(1).strip() if h1 else path.stem
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
    )
