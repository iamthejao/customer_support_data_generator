"""Prompt template registry with content-hash versioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from csfd.utils.hashing import short_hash


@dataclass(slots=True, frozen=True)
class PromptHandle:
    name: str
    template: jinja2.Template
    source: str
    version: str


@dataclass(slots=True)
class PromptRegistry:
    root: Path
    _handles: dict[str, PromptHandle] = field(default_factory=dict)
    _env: jinja2.Environment | None = None

    def load(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.root),
            keep_trailing_newline=True,
            autoescape=False,  # templates are markdown/prompts, not HTML
        )
        self._handles.clear()
        for path in self.root.rglob("*.md.j2"):
            rel = path.relative_to(self.root).with_suffix("").with_suffix("")
            name = ".".join(rel.parts)
            source = path.read_text(encoding="utf-8")
            template = self._env.from_string(source)
            self._handles[name] = PromptHandle(
                name=name,
                template=template,
                source=source,
                version=short_hash(source),
            )

    def get(self, name: str) -> PromptHandle:
        if not self._handles:
            raise RuntimeError("Registry not loaded; call .load() first")
        if name not in self._handles:
            raise KeyError(name)
        return self._handles[name]

    def all(self) -> dict[str, PromptHandle]:
        return dict(self._handles)
