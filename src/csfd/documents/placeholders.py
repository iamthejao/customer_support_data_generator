"""Identifier placeholders in document text, resolved by code.

Builders (and, later, model-written prose) never type a part number, error
code, document number or section number themselves. They write a placeholder
that code resolves against the identifier registry and the document's own
sections, so a number can never be mistyped or invented:

* ``{part:KD-60-2204}`` -> ``KD-60-2204 Thermocouple, type S``
* ``{error:E-21}`` -> ``E-21 (Heating element open circuit)``
* ``{doc:service_manual}`` -> ``KD-SM-CF600-EN rev. C`` (the current revision)
* ``{section:sec-5.3.1}`` -> ``section 5.3.1``

A placeholder naming something the registry or the document lacks is an error.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from csfd.documents.ir import (
    Admonition,
    Block,
    DocumentIR,
    ItemList,
    Paragraph,
    Procedure,
    Section,
    Table,
)
from csfd.documents.registry import ProductFacts

PLACEHOLDER = re.compile(r"\{(part|error|doc|section):([^{}]+)\}")


def resolve_text(text: str, facts: ProductFacts | None, doc: DocumentIR) -> str:
    """``text`` with every placeholder replaced; ``facts`` is the document's model entry."""

    def sub(m: re.Match[str]) -> str:
        kind, key = m.group(1), m.group(2).strip()
        if kind == "section":
            try:
                return f"section {doc.section(key).number}"
            except KeyError as exc:
                raise ValueError(f"document {doc.doc_id}: no section {key!r}") from exc
        if facts is None:
            raise ValueError(f"document {doc.doc_id}: {m.group()} needs a registry entry")
        if kind == "part":
            part = next((p for p in facts.parts if p.part_number == key), None)
            if part is None:
                raise ValueError(f"{key!r} is not a registry part number of the {facts.model}")
            return part.entry
        if kind == "error":
            code = next((c for c in facts.error_codes if c.code == key), None)
            if code is None:
                raise ValueError(f"{key!r} is not a registry error code of the {facts.model}")
            return f"{code.code} ({code.meaning})"
        ref = next((d for d in facts.documents if d.doc_type == key), None)
        if ref is None:
            raise ValueError(f"the {facts.model} has no registry document {key!r}")
        return f"{ref.doc_number} rev. {ref.revisions[-1]}"

    return PLACEHOLDER.sub(sub, text)


def _block(block: Block, r: Callable[[str], str]) -> Block:
    if isinstance(block, Paragraph | Admonition):
        return block.model_copy(update={"text": r(block.text)})
    if isinstance(block, ItemList):
        return block.model_copy(update={"items": [r(i) for i in block.items]})
    if isinstance(block, Table):
        return block.model_copy(
            update={
                "caption": r(block.caption),
                "rows": [[r(cell) for cell in row] for row in block.rows],
            }
        )
    if isinstance(block, Procedure):
        steps = [
            st.model_copy(
                update={"text": r(st.text), "expected": r(st.expected) if st.expected else None}
            )
            for st in block.steps
        ]
        return block.model_copy(update={"title": r(block.title), "steps": steps})
    return block


def map_text(doc: DocumentIR, r: Callable[[str], str]) -> DocumentIR:
    """``doc`` with ``r`` applied to its title, every section title and every block's text."""

    def section(s: Section) -> Section:
        return s.model_copy(
            update={
                "title": r(s.title),
                "blocks": [_block(b, r) for b in s.blocks],
                "subsections": [section(c) for c in s.subsections],
            }
        )

    return doc.model_copy(
        update={"title": r(doc.title), "sections": [section(s) for s in doc.sections]}
    )


def resolve_document(doc: DocumentIR, facts: ProductFacts | None) -> DocumentIR:
    """``doc`` with the placeholders of every title and block resolved."""
    return map_text(doc, lambda text: resolve_text(text, facts, doc))
