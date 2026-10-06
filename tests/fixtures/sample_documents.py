"""A fixture document builder: a CF-600 service-manual extract plus one KB article per problem.

Stands in for the real builders (not shipped yet) so the document store,
renderers, sidecars and retrieval export can be tested end to end. Everything
is deterministic and needs no model.
"""

from __future__ import annotations

from datetime import date

from csfd.documents.build import (
    BuildContext,
    BuildResult,
    BuiltDocument,
    CaseAnswer,
    DocumentLink,
)
from csfd.documents.ir import (
    Admonition,
    AppliesTo,
    Asset,
    DocumentIR,
    Figure,
    ItemList,
    Paragraph,
    Procedure,
    Revision,
    Section,
    Step,
    Table,
    make_doc_id,
)

SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="200" viewBox="0 0 320 200">
<rect x="0" y="0" width="320" height="200" fill="white"/>
<rect x="60" y="30" width="160" height="110" rx="8" fill="none" stroke="black" stroke-width="2"/>
<circle cx="260" cy="60" r="12" fill="white" stroke="black"/>
<line x1="200" y1="70" x2="248" y2="60" stroke="black"/>
<rect x="150" y="160" width="160" height="30" fill="none" stroke="black"/>
</svg>
"""


def figure_asset() -> Asset:
    return Asset.from_bytes(
        SVG, media_type="image/svg+xml", origin="template", provenance={"template": "fixture"}
    )


def service_manual(
    company: str, *, revision: str = "C", issued: date = date(2025, 9, 15)
) -> DocumentIR:
    asset = figure_asset()
    return DocumentIR(
        doc_id=make_doc_id(company, "KD-SM-CF600-EN", "en", revision),
        doc_type="service_manual",
        tier="library",
        doc_number="KD-SM-CF600-EN",
        title="CF-600 Ceramic Firing Furnace",
        subtitle="Service Manual (extract)",
        company=company,
        revision=revision,
        issue_date=issued,
        audience="internal",
        status="current" if revision == "C" else "superseded",
        applies_to=AppliesTo(models=["CF-600"], serials=["CF600-0001-AA to CF600-9999-ZZ"]),
        revisions=[
            Revision(
                rev="B",
                issued=date(2024, 6, 12),
                description="Added W-07",
                author="Technical Service",
            ),
            Revision(
                rev=revision,
                issued=issued,
                description="Silver test revised",
                author="Technical Service",
            ),
        ],
        sections=[
            Section(
                id="sec-1",
                number="1",
                title="Safety",
                blocks=[
                    Admonition(
                        id="sec-1-b1", level="warning", text="The chamber reaches 1,200 °C."
                    ),
                    Admonition(
                        id="sec-1-b2", level="caution", text="Disconnect mains power first."
                    ),
                ],
            ),
            Section(
                id="sec-3",
                number="3",
                title="Main components",
                blocks=[
                    Figure(
                        id="fig-3-1",
                        asset_id=asset.asset_id,
                        caption="Figure 3-1. CF-600 front view with item numbers",
                        alt="Line drawing of the CF-600 front with balloon 1 on the thermocouple.",
                    ),
                    ItemList(id="sec-3-b2", items=["Firing chamber", "Thermocouple, type S"]),
                ],
            ),
            Section(
                id="sec-6",
                number="6",
                title="Calibration",
                subsections=[
                    Section(
                        id="sec-6.2",
                        number="6.2",
                        title="Silver test",
                        covers=["temperature calibration drift"],
                        blocks=[
                            Paragraph(id="sec-6.2-b1", text="Pure silver melts at 961 °C."),
                            Procedure(
                                id="proc-6.2",
                                title="Silver test",
                                steps=[
                                    Step(id="proc-6.2-s1", text="Place the sample in the centre."),
                                    Step(
                                        id="proc-6.2-s2",
                                        text="Start Service > Calibration > Silver test.",
                                        expected="The display shows the melt temperature.",
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            ),
            Section(
                id="sec-9",
                number="9",
                title="Spare parts",
                blocks=[
                    Table(
                        id="tbl-9-1",
                        caption="Table 9-1. Spare parts",
                        columns=["Item", "Part number", "Description"],
                        rows=[["1", "KD-60-2204", "Thermocouple, type S | 0.35 mm"]],
                        role="parts",
                    )
                ],
            ),
        ],
    )


def kb_article(company: str, problem_id: str, index: int) -> DocumentIR:
    return DocumentIR(
        doc_id=make_doc_id(company, f"KB-{index:04d}", "en", "A"),
        doc_type="kb_article",
        tier="problem",
        doc_number=f"KB-{index:04d}",
        title=f"Troubleshooting article {index}",
        company=company,
        revision="A",
        issue_date=date(2025, 11, 3),
        audience="internal",
        problem_id=problem_id,
        sections=[
            Section(
                id="sec-1",
                number="1",
                title="Checks",
                blocks=[Paragraph(id="sec-1-b1", text="Run the checks in order.")],
            ),
            Section(
                id="sec-2",
                number="2",
                title="Cause: loose connector",
                covers=["loose connector"],
                blocks=[Paragraph(id="sec-2-b1", text="Reseat the connector.")],
            ),
            Section(
                id="sec-3",
                number="3",
                title="Cause: failing supply",
                covers=["failing supply"],
                blocks=[Paragraph(id="sec-3-b1", text="Replace the supply module.")],
            ),
        ],
    )


class SampleBuilder:
    """Library manual (current + superseded revision) and a KB article per problem."""

    name = "sample"

    def build(self, ctx: BuildContext) -> BuildResult:
        company = ctx.company_name
        current = service_manual(company)
        older = service_manual(company, revision="B", issued=date(2024, 6, 12))
        result = BuildResult(
            documents=[
                BuiltDocument(document=current, assets=[figure_asset()]),
                BuiltDocument(document=older, assets=[figure_asset()]),
            ]
        )
        articles: dict[str, DocumentIR] = {}
        for i, problem in enumerate(ctx.problems):
            article = kb_article(company, problem.id, i)
            articles[problem.id] = article
            result.documents.append(BuiltDocument(document=article))
        for case in ctx.cases:
            article = articles[case.problem_id]
            result.links += [
                DocumentLink(
                    case_uid=case.case_uid,
                    doc_id=article.doc_id,
                    section_id="sec-2",
                    relation="resolves",
                    grade=2,
                    basis="root_cause_fix",
                ),
                DocumentLink(
                    case_uid=case.case_uid,
                    doc_id=article.doc_id,
                    section_id="sec-1",
                    relation="supports",
                    grade=1,
                    basis="check:1",
                ),
                DocumentLink(
                    case_uid=case.case_uid,
                    doc_id=article.doc_id,
                    section_id="sec-3",
                    relation="hard_negative",
                    grade=0,
                    basis="other_cause",
                ),
                DocumentLink(
                    case_uid=case.case_uid,
                    round_index=1,
                    doc_id=current.doc_id,
                    section_id="sec-6.2",
                    relation="supports",
                    grade=1,
                    basis="check:2",
                ),
                DocumentLink(
                    case_uid=case.case_uid,
                    doc_id=older.doc_id,
                    section_id="sec-6.2",
                    relation="hard_negative",
                    grade=0,
                    basis="superseded_revision",
                ),
            ]
            result.cases.append(CaseAnswer(case_uid=case.case_uid, answer_source="documents"))
        return result
