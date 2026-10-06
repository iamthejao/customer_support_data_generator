"""The document IR: ids, validation, traversal."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from csfd.documents.ir import (
    Asset,
    DocumentIR,
    Paragraph,
    Procedure,
    Section,
    Step,
    Table,
    make_doc_id,
)
from tests.fixtures.sample_documents import SVG, kb_article, service_manual


def _doc(**overrides: object) -> DocumentIR:
    fields: dict[str, object] = {
        "doc_id": "acme:D-1:en:rev-A",
        "doc_type": "kb_article",
        "tier": "library",
        "doc_number": "D-1",
        "title": "T",
        "company": "Acme",
        "revision": "A",
        "issue_date": date(2025, 1, 1),
        "audience": "internal",
        "sections": [Section(id="sec-1", number="1", title="One")],
    }
    fields.update(overrides)
    return DocumentIR.model_validate(fields)


def test_doc_id_is_run_independent_and_slugs_the_company() -> None:
    assert make_doc_id("Kalvora Dental", "KD-SM-CF600-EN", "en", "C") == (
        "kalvora-dental:KD-SM-CF600-EN:en:rev-C"
    )


def test_walk_yields_sections_in_reading_order_with_depth_and_parents() -> None:
    doc = service_manual("Kalvora Dental")
    walked = [(s.id, depth, [p.id for p in parents]) for s, depth, parents in doc.walk()]
    assert walked == [
        ("sec-1", 1, []),
        ("sec-3", 1, []),
        ("sec-6", 1, []),
        ("sec-6.2", 2, ["sec-6"]),
        ("sec-9", 1, []),
    ]
    assert doc.section("sec-6.2").title == "Silver test"
    with pytest.raises(KeyError):
        doc.section("sec-404")


def test_folder_name_is_number_and_revision() -> None:
    assert service_manual("Kalvora Dental").folder_name == "KD-SM-CF600-EN_revC"
    assert _doc(language="de").folder_name == "D-1_revA_de"


def test_duplicate_ids_across_sections_blocks_and_steps_are_rejected() -> None:
    clash = Section(
        id="sec-1",
        number="1",
        title="One",
        blocks=[
            Procedure(id="proc-1", title="P", steps=[Step(id="sec-1", text="step")]),
        ],
    )
    with pytest.raises(ValidationError, match="duplicate id 'sec-1'"):
        _doc(sections=[clash])


def test_ids_must_be_lower_case_anchors() -> None:
    with pytest.raises(ValidationError, match="invalid id"):
        _doc(sections=[Section(id="Sec 1", number="1", title="One")])


def test_tables_must_be_rectangular() -> None:
    with pytest.raises(ValidationError, match="row 0 has 1 cells, expected 2"):
        Table(id="t-1", caption="c", columns=["a", "b"], rows=[["x"]])


def test_problem_and_case_tiers_need_their_owner() -> None:
    with pytest.raises(ValidationError, match="needs problem_id"):
        _doc(tier="problem")
    with pytest.raises(ValidationError, match="needs case_uid"):
        _doc(tier="case")
    assert kb_article("Acme", "run:p:0000", 0).problem_id == "run:p:0000"


def test_asset_id_is_the_content_hash() -> None:
    asset = Asset.from_bytes(SVG, media_type="image/svg+xml", origin="template")
    assert len(asset.asset_id) == 64
    with pytest.raises(ValidationError, match="sha256"):
        Asset(asset_id="0" * 64, media_type="image/png", content=b"x", origin="template")


def test_ir_round_trips_through_json() -> None:
    doc = service_manual("Kalvora Dental")
    again = DocumentIR.model_validate_json(doc.model_dump_json())
    assert again == doc
    assert isinstance(again.section("sec-6.2").blocks[0], Paragraph)
    assert again.section("sec-6.2").covers == ["temperature calibration drift"]



def test_section_ids_must_fit_a_word_bookmark() -> None:
    with pytest.raises(ValidationError, match="too long for a Word bookmark"):
        _doc(sections=[Section(id="sec-" + "1" * 40, number="1", title="Long")])
