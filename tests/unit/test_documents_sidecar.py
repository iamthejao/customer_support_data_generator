"""Markdown and JSON sidecars: anchors, chunks, figures."""

from __future__ import annotations

from csfd.documents.sidecar import chunks, render_markdown, sidecar
from tests.fixtures.sample_documents import figure_asset, service_manual

DOC = service_manual("Kalvora Dental")
FIGS = {"fig-3-1": "assets/fig-3-1.png"}


def test_markdown_headings_carry_section_anchors() -> None:
    md = render_markdown(DOC, FIGS)
    assert md.startswith("# CF-600 Ceramic Firing Furnace\n")
    assert "## 6 Calibration {#sec-6}" in md
    assert "### 6.2 Silver test {#sec-6.2}" in md
    assert "> **WARNING:** The chamber reaches 1,200 °C." in md
    assert "2. Start Service > Calibration > Silver test. *Result: The display shows" in md
    assert "![Figure 3-1. CF-600 front view with item numbers](assets/fig-3-1.png)" in md
    # A pipe inside a cell is escaped so the table keeps its columns.
    assert "| 1 | KD-60-2204 | Thermocouple, type S \\| 0.35 mm |" in md


def test_one_chunk_per_section_with_its_own_text_and_heading_path() -> None:
    out = chunks(DOC, FIGS, {"sec-6.2": (2, 3)})
    assert [c.section_id for c in out] == ["sec-1", "sec-3", "sec-6", "sec-6.2", "sec-9"]
    silver = out[3]
    assert silver.chunk_id == f"{DOC.doc_id}#sec-6.2"
    assert silver.heading_path == ["6 Calibration", "6.2 Silver test"]
    assert silver.depth == 2
    assert silver.pdf_pages == (2, 3)
    assert silver.docx_bookmark == "sec_6_2"
    assert silver.covers == ["temperature calibration drift"]
    assert silver.text.startswith("Pure silver melts at 961 °C.")
    # The parent's chunk holds only its own blocks (none here), not the subsection's.
    assert out[2].text == ""
    assert out[0].pdf_pages is None


def test_json_sidecar_lists_figures_with_provenance() -> None:
    asset = figure_asset()
    out = sidecar(
        DOC,
        chunks(DOC, FIGS),
        {asset.asset_id: asset},
        {"fig-3-1": ["assets/fig-3-1.svg", "assets/fig-3-1.png"]},
        {"x.docx": "abc"},
    )
    assert out["schema"] == "csfd.document/1"
    assert out["document"]["doc_id"] == DOC.doc_id
    assert len(out["chunks"]) == 5
    [fig] = out["figures"]
    assert fig["section_id"] == "sec-3"
    assert fig["files"] == ["assets/fig-3-1.svg", "assets/fig-3-1.png"]
    assert fig["origin"] == "template"
    assert fig["alt"].startswith("Line drawing")
    assert out["files"] == {"x.docx": "abc"}
