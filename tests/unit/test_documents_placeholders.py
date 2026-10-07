"""Identifier placeholders are resolved from the registry and the document, never guessed."""

from __future__ import annotations

from datetime import date

import pytest

from csfd.documents.ir import DocumentIR, Paragraph, Section
from csfd.documents.placeholders import resolve_document, resolve_text
from csfd.documents.registry import DocRef, ErrorCode, Part, ProductFacts

FACTS = ProductFacts(
    model="CF-600",
    source="seed",
    part_number_format="KD-60-####",
    parts=[Part(part_number="KD-60-2204", name="Thermocouple, type S", item=3)],
    error_codes=[ErrorCode(code="E-21", meaning="Heating element open circuit")],
    documents=[
        DocRef(
            doc_number="KD-SM-CF600-EN",
            doc_type="service_manual",
            revisions=["A", "B"],
            issued=[date(2024, 1, 1), date(2025, 1, 1)],
        )
    ],
)


def _doc(text: str) -> DocumentIR:
    return DocumentIR(
        doc_id="acme:KB-1:en:rev-A",
        doc_type="kb_article",
        tier="library",
        doc_number="KB-1",
        title="t",
        company="Acme",
        revision="A",
        issue_date=date(2025, 1, 1),
        audience="internal",
        sections=[
            Section(id="sec-1", number="1", title="One", blocks=[Paragraph(id="p1", text=text)]),
            Section(id="sec-2.4", number="2.4", title="Two"),
        ],
    )


def test_every_kind_resolves() -> None:
    text = (
        "Fit {part:KD-60-2204} if {error:E-21} shows; see {doc:service_manual}, {section:sec-2.4}."
    )
    assert resolve_text(text, FACTS, _doc("")) == (
        "Fit KD-60-2204 Thermocouple, type S if E-21 (Heating element open circuit) shows; "
        "see KD-SM-CF600-EN rev. B, section 2.4."
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{part:KD-60-9999}", "not a registry part number"),
        ("{error:E-99}", "not a registry error code"),
        ("{doc:datasheet}", "no registry document"),
        ("{section:sec-9}", "no section"),
    ],
)
def test_an_unknown_identifier_is_an_error(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_text(text, FACTS, _doc(""))


def test_registry_placeholders_need_a_registry_entry() -> None:
    assert resolve_text("{section:sec-1}", None, _doc("")) == "section 1"
    with pytest.raises(ValueError, match="needs a registry entry"):
        resolve_text("{part:KD-60-2204}", None, _doc(""))


def test_resolve_document_rewrites_block_text() -> None:
    doc = resolve_document(_doc("Replace {part:KD-60-2204}."), FACTS)
    block = doc.section("sec-1").blocks[0]
    assert isinstance(block, Paragraph)
    assert block.text == "Replace KD-60-2204 Thermocouple, type S."
