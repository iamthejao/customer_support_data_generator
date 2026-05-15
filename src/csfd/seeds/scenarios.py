"""Parse scenarios_seed.md into a typed ScenarioCatalogue."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
# Match hyphen, en-dash (U+2013), or em-dash (U+2014)
_CATEGORY_SEP = re.compile(r"\s*" + "[-–—]" + r"\s*")  # noqa: RUF001


@dataclass(slots=True)
class Scenario:
    category: str
    title: str
    summary: str


@dataclass(slots=True)
class ScenarioCatalogue:
    scenarios: list[Scenario] = field(default_factory=list)
    source_path: Path = field(default_factory=lambda: Path("."))


def parse_scenarios_seed(path: Path) -> ScenarioCatalogue:
    text = path.read_text(encoding="utf-8")
    headings = list(_H2.finditer(text))
    items: list[Scenario] = []
    for i, m in enumerate(headings):
        heading = m.group(1).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        parts = _CATEGORY_SEP.split(heading, maxsplit=1)
        if len(parts) == 2:
            category, title = parts[0].strip(), parts[1].strip()
        else:
            category, title = heading, heading
        items.append(Scenario(category=category, title=title, summary=body))
    return ScenarioCatalogue(scenarios=items, source_path=path)
