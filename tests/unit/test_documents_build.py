"""The builder interface: registry, merge, and the checks on what builders return."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from csfd.documents.build import (
    BuildContext,
    BuildResult,
    BuiltDocument,
    CaseAnswer,
    CaseInput,
    DocumentLink,
    resolve_builders,
    run_builders,
)
from csfd.storage.repository import ProblemRecord
from tests.fixtures.sample_documents import (
    SampleBuilder,
    figure_asset,
    kb_article,
    service_manual,
)


def _ctx(**overrides: object) -> BuildContext:
    problem = ProblemRecord(
        id="run:p:0000",
        run_id="run",
        title="t",
        summary="s",
        background="b",
        category="c",
        complexity="simple",
        resolution_hints={},
        quality_flag=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    fields: dict[str, object] = {
        "run_id": "run",
        "run_seed": 7,
        "company_name": "Kalvora Dental",
        "problems": [problem],
        "cases": [
            CaseInput(
                case_uid="run:000000",
                slot_index=0,
                problem_id="run:p:0000",
                ticket_type="l1",
                customer_tier="standard",
            )
        ],
    }
    fields.update(overrides)
    return BuildContext.model_validate(fields)


class _Fixed:
    def __init__(self, name: str, result: BuildResult) -> None:
        self.name = name
        self._result = result

    def build(self, ctx: BuildContext) -> BuildResult:
        return self._result


def test_link_grade_must_fit_its_relation() -> None:
    DocumentLink(case_uid="c", doc_id="d", relation="equivalent", grade=2, basis="same_cause")
    with pytest.raises(ValidationError, match="cannot have grade"):
        DocumentLink(case_uid="c", doc_id="d", relation="resolves", grade=1, basis="x")
    with pytest.raises(ValidationError, match="cannot have grade"):
        DocumentLink(case_uid="c", doc_id="d", relation="hard_negative", grade=1, basis="x")


def test_coverage_defaults_to_documented() -> None:
    ctx = _ctx(coverage={"run:p:0000": {"loose connector": "agent_only"}})
    assert ctx.coverage_of("run:p:0000", "loose connector") == "agent_only"
    assert ctx.coverage_of("run:p:0000", "failing supply") == "documented"
    assert ctx.coverage_of("other", "anything") == "documented"


def test_resolve_builders_names_the_known_ones_on_a_typo() -> None:
    registry = {"sample": SampleBuilder}
    assert [b.name for b in resolve_builders(["sample"], registry)] == ["sample"]
    with pytest.raises(
        ValueError, match=r"unknown document builder\(s\): smaple \(known: sample\)"
    ):
        resolve_builders(["smaple"], registry)
    with pytest.raises(ValueError, match="none registered"):
        resolve_builders(["x"], {})


def test_run_builders_merges_and_records_who_made_each_document() -> None:
    result, made_by = run_builders(_ctx(), [SampleBuilder()])
    assert len(result.documents) == 3
    assert set(made_by.values()) == {"sample"}
    assert {c.answer_source for c in result.cases} == {"documents"}


def test_a_document_built_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="built twice"):
        run_builders(_ctx(), [SampleBuilder(), SampleBuilder()])


def test_links_must_point_at_known_cases_documents_and_sections() -> None:
    doc = kb_article("Kalvora Dental", "run:p:0000", 0)
    built = [BuiltDocument(document=doc)]

    def link(**kw: object) -> DocumentLink:
        fields: dict[str, object] = {
            "case_uid": "run:000000",
            "doc_id": doc.doc_id,
            "section_id": "sec-2",
            "relation": "resolves",
            "grade": 2,
            "basis": "x",
        }
        fields.update(kw)
        return DocumentLink.model_validate(fields)

    for bad, message in [
        (link(case_uid="nope"), "unknown case"),
        (link(doc_id="nope"), "unknown document"),
        (link(section_id="sec-9"), "unknown section"),
    ]:
        with pytest.raises(ValueError, match=message):
            run_builders(_ctx(), [_Fixed("f", BuildResult(documents=built, links=[bad]))])


def test_a_figure_needs_its_image() -> None:
    built = BuiltDocument(document=service_manual("Kalvora Dental"))  # no assets
    with pytest.raises(ValueError, match="figure fig-3-1 has no asset"):
        run_builders(_ctx(), [_Fixed("f", BuildResult(documents=[built]))])


def test_agent_only_causes_must_not_be_documented() -> None:
    doc = kb_article("Kalvora Dental", "run:p:0000", 0)
    ctx = _ctx(coverage={"run:p:0000": {"loose connector": "agent_only"}})
    result = BuildResult(documents=[BuiltDocument(document=doc)])
    with pytest.raises(ValueError, match="agent knowledge only"):
        run_builders(ctx, [_Fixed("f", result)])


def test_shared_and_case_documents_must_not_document_agent_only_causes() -> None:
    # The library manual documents "temperature calibration drift" (section 6.2); the
    # match ignores case and punctuation.
    manual = BuiltDocument(document=service_manual("Kalvora Dental"), assets=[figure_asset()])
    ctx = _ctx(coverage={"run:p:0000": {"Temperature calibration-drift": "agent_only"}})
    with pytest.raises(ValueError, match=r"sec-6\.2 documents"):
        run_builders(ctx, [_Fixed("f", BuildResult(documents=[manual]))])
    log = kb_article("Kalvora Dental", "run:p:0000", 0).model_copy(
        update={"tier": "case", "problem_id": None, "case_uid": "run:000000"}
    )
    ctx = _ctx(coverage={"run:p:0000": {"loose connector": "agent_only"}})
    with pytest.raises(ValueError, match="agent knowledge only"):
        run_builders(ctx, [_Fixed("f", BuildResult(documents=[BuiltDocument(document=log)]))])
    # A documented or partial cause is fine.
    ctx = _ctx(coverage={"run:p:0000": {"temperature calibration drift": "partial"}})
    result, _ = run_builders(ctx, [_Fixed("f", BuildResult(documents=[manual]))])
    assert len(result.documents) == 1


def test_one_answer_source_per_known_case() -> None:
    twice = BuildResult(
        cases=[
            CaseAnswer(case_uid="run:000000", answer_source="documents"),
            CaseAnswer(case_uid="run:000000", answer_source="partial"),
        ]
    )
    with pytest.raises(ValueError, match="more than one answer source"):
        run_builders(_ctx(), [_Fixed("f", twice)])
    unknown = BuildResult(cases=[CaseAnswer(case_uid="nope", answer_source="agent_knowledge")])
    with pytest.raises(ValueError, match="unknown case"):
        run_builders(_ctx(), [_Fixed("f", unknown)])
