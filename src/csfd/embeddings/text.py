"""Single source of truth for the text fed into the embedding model.

Both Phase 1 generation and any downstream re-embedding must call this so the
stored vectors and the live candidate vectors derive from identical strings.
"""

from __future__ import annotations

from typing import Literal, Protocol

TextTemplate = Literal[
    "title_summary",
    "title_summary_background",
    "title_summary_symptoms_root_cause",
]


class _ProblemLike(Protocol):
    title: str
    summary: str
    background: str
    symptoms: list[str]
    root_cause: list[str]


def text_for_problem(p: _ProblemLike, template: TextTemplate) -> str:
    if template == "title_summary":
        return f"{p.title}\n\n{p.summary}"
    if template == "title_summary_background":
        return f"{p.title}\n\n{p.summary}\n\n{p.background}"
    if template == "title_summary_symptoms_root_cause":
        parts = [f"{p.title}\n\n{p.summary}"]
        if p.symptoms:
            parts.append("Symptoms:\n" + "\n".join(f"- {s}" for s in p.symptoms))
        if p.root_cause:
            parts.append("Root cause:\n" + "\n".join(f"- {c}" for c in p.root_cause))
        return "\n\n".join(parts)
    raise ValueError(f"unknown template: {template!r}")
