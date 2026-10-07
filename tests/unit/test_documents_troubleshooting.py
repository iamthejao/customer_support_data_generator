"""Troubleshooting documents: KB articles and service-manual chapters built from diagnosis plans."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from csfd.diagnosis import CandidateCause, DiagnosisPlan, DiagnosticCheck, guide_causes
from csfd.documents.build import (
    BuildContext,
    BuildResult,
    CaseInput,
    DocumentLink,
    run_builders,
)
from csfd.documents.coverage import Coverage
from csfd.documents.ir import DocumentIR
from csfd.documents.placeholders import PLACEHOLDER
from csfd.documents.registry import ProductFacts, build_registry, kb_documents
from csfd.documents.sidecar import render_markdown
from csfd.documents.troubleshooting import TroubleshootingBuilder
from csfd.seeds.company import CompanyProfile, parse_company_seed
from csfd.storage.repository import ProblemRecord

SEED = Path(__file__).resolve().parents[1] / "fixtures" / "registry_company_seed.md"
START = datetime(2026, 3, 1, tzinfo=UTC)
RUN = "run"
P0, P1 = f"{RUN}:p:0000", f"{RUN}:p:0001"


@pytest.fixture(scope="module")
def company() -> CompanyProfile:
    return parse_company_seed(SEED)


@pytest.fixture(scope="module")
def registry(company: CompanyProfile) -> dict[str, ProductFacts]:
    return build_registry(company, seed=1, calendar_start=START, names={})


def _fan(registry: dict[str, ProductFacts]) -> str:
    return next(p.entry for p in registry["AX-100"].parts if p.name == "Fan tray, front")


def _plans(registry: dict[str, ProductFacts]) -> dict[str, DiagnosisPlan]:
    fan = CandidateCause(
        cause="Failed fan tray",
        resolution_steps=["Replace the fan tray"],
        parts=[_fan(registry)],
        verification="Watch the fan status LED for five minutes",
        verification_finding="the LED stays green",
    )
    p0 = DiagnosisPlan(
        candidate_causes=[
            fan.model_copy(update={"is_root_cause": True}),
            CandidateCause(
                cause="Blocked air inlet",
                resolution_steps=["Clean the air inlet filter"],
                workaround="Run the unit at reduced load",
            ),
            CandidateCause(
                cause="Faulty power supply",
                resolution_steps=["Replace the power supply"],
                parts=["AC-PSU-001 Power supply unit, 240 W", "PSU cable"],
            ),
        ],
        checks=[
            DiagnosticCheck(
                check="Fan LED check",
                how_to_check="Look at the fan LED on the front panel",
                finding="the LED blinks red",
                rules_out=["Blocked air inlet", "Faulty power supply"],
                confirms_cause=True,
            )
        ],
        safety=["Disconnect mains power before opening the cover"],
    )
    p1 = DiagnosisPlan(
        candidate_causes=[
            fan,
            CandidateCause(
                cause="Dust in the heat sink",
                is_root_cause=True,
                resolution_steps=["Blow the heat sink clean with dry air"],
            ),
        ],
        checks=[
            DiagnosticCheck(
                check="Heat sink check",
                how_to_check="Shine a torch through the side vents",
                finding="grey dust covers the fins",
                confirms_cause=True,
            )
        ],
    )
    return {P0: p0, P1: p1}


def _problem(pid: str, plan: DiagnosisPlan, title: str, symptoms: list[str]) -> ProblemRecord:
    return ProblemRecord(
        id=pid,
        run_id=RUN,
        title=title,
        summary="The appliance shuts down under load.",
        background="b",
        category="cooling",
        complexity="medium",
        resolution_hints={},
        quality_flag=None,
        created_at=START,
        symptoms=symptoms,
        root_cause=["SECRET root cause text"],
        diagnosis_plan=plan.model_dump(),
    )


def _ctx(
    company: CompanyProfile,
    registry: dict[str, ProductFacts],
    coverage: dict[str, dict[str, Coverage]] | None = None,
) -> BuildContext:
    plans = _plans(registry)
    problems = [
        _problem(P0, plans[P0], "Appliance overheats", ["The display shows F-01", "It runs hot"]),
        _problem(P1, plans[P1], "Appliance throttles", ["Throughput drops in the afternoon"]),
    ]
    cases = [
        CaseInput(
            case_uid=f"{RUN}:00000{i}",
            slot_index=i,
            problem_id=pid,
            ticket_type="l1",
            customer_tier="standard",
        )
        for i, pid in enumerate([P0, P1])
    ]
    return BuildContext(
        run_id=RUN,
        run_seed=7,
        company_name=company.name,
        company=company,
        calendar_start=START,
        problems=problems,
        cases=cases,
        coverage=coverage or {},
        registry=registry,
        problem_models={P0: "AX-100", P1: "AX-100"},
    )


def _build(ctx: BuildContext) -> BuildResult:
    result, _ = run_builders(ctx, [TroubleshootingBuilder()])
    return result


def _docs(result: BuildResult) -> dict[str, DocumentIR]:
    return {
        b.document.doc_type + ":" + (b.document.problem_id or ""): b.document
        for b in result.documents
    }


def _text(doc: DocumentIR) -> str:
    return render_markdown(doc, {})


def _links(result: BuildResult, case: int) -> list[DocumentLink]:
    return [link for link in result.links if link.case_uid == f"{RUN}:00000{case}"]


def _graded(links: list[DocumentLink], doc: DocumentIR) -> dict[str, tuple[str, int]]:
    return {
        link.section_id or "": (link.relation, link.grade)
        for link in links
        if link.doc_id == doc.doc_id
    }


def test_one_kb_article_per_problem_and_one_chapter_per_model(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    result = _build(_ctx(company, registry))
    docs = _docs(result)
    assert sorted(docs) == ["kb_article:" + P0, "kb_article:" + P1, "service_manual:"]
    kb = docs["kb_article:" + P0]
    assert kb.tier == "problem" and kb.audience == "internal"
    assert kb.applies_to.models == ["AX-100"]
    refs = kb_documents(company.name, [P0, P1], seed=7, calendar_start=START)
    assert kb.doc_number == refs[P0].doc_number and kb.revision == refs[P0].revisions[-1]
    assert kb.issue_date < START.date()

    chapter = docs["service_manual:"]
    sm = next(d for d in registry["AX-100"].documents if d.doc_type == "service_manual")
    assert chapter.tier == "library"
    assert (chapter.doc_number, chapter.revision) == (sm.doc_number, sm.revisions[-1])
    assert [r.rev for r in chapter.revisions] == sm.revisions
    assert [s.id for s in chapter.sections] == ["sec-5"]
    assert [s.title for s in chapter.sections[0].subsections] == [
        "Error and warning codes",
        "Symptoms, causes and checks",
        "Remedies",
    ]


def test_causes_follow_the_agent_guide_order_and_nothing_gives_the_answer_away(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    ctx = _ctx(company, registry)
    result = _build(ctx)
    kb = _docs(result)["kb_article:" + P0]
    plan = _plans(registry)[P0]
    expected = [c.cause for c in guide_causes(plan, seed_label=f"{ctx.run_seed}:{P0}")]
    remedies = kb.section("sec-5").subsections
    assert [s.title for s in remedies] == expected
    for doc in (b.document for b in result.documents):
        text = _text(doc).lower()
        assert "root cause" not in text
        assert "secret" not in text
        assert "blinks red" not in text  # a check's finding is the customer's to report
        assert "grey dust" not in text


def test_parts_error_codes_and_document_numbers_come_from_the_registry(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    result = _build(_ctx(company, registry))
    kb = _docs(result)["kb_article:" + P0]
    text = _text(kb)
    fan_number = _fan(registry).split()[0]
    assert f"| {fan_number} | Fan tray, front |" in text
    assert "| AC-PSU-001 | Power supply unit, 240 W | 4 |" in text
    assert "| - | PSU cable | - |" in text  # not in the registry: named, never numbered
    assert "| F-01 | Fan tray failure | Replace the fan tray |" in text  # the symptom's code
    for ref in registry["AX-100"].documents:
        assert f"{ref.doc_number} rev. {ref.revisions[-1]}" in text
    assert not PLACEHOLDER.search(text)  # every placeholder resolved
    chapter = _docs(result)["service_manual:"]
    for code in registry["AX-100"].error_codes:
        assert f"| {code.code} |" in _text(chapter)


def test_documented_root_cause_links(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    result = _build(_ctx(company, registry))
    docs = _docs(result)
    kb0, kb1, chapter = docs["kb_article:" + P0], docs["kb_article:" + P1], docs["service_manual:"]
    links = _links(result, 0)
    fan = kb0.section(_section_titled(kb0, "Failed fan tray"))
    assert _graded(links, kb0) == {
        "sec-4": ("supports", 1),  # possible causes and the checks
        "sec-2": ("supports", 1),  # the symptom's error code
        fan.id: ("resolves", 2),
        _section_titled(kb0, "Blocked air inlet"): ("hard_negative", 0),
        _section_titled(kb0, "Faulty power supply"): ("hard_negative", 0),
    }
    graded = _graded(links, chapter)
    assert graded[_section_titled(chapter, "Failed fan tray")] == ("resolves", 2)
    assert graded["sec-5.1"] == ("supports", 1)  # error codes table
    assert graded[_section_titled(chapter, "Appliance overheats")] == ("supports", 1)
    assert graded[_section_titled(chapter, "Blocked air inlet")] == ("hard_negative", 0)
    # The other problem's article documents the same cause: an equivalent answer.
    assert _graded(links, kb1) == {_section_titled(kb1, "Failed fan tray"): ("equivalent", 2)}
    assert all(link.cause == "Failed fan tray" for link in links if link.grade == 2)
    assert [(c.case_uid, c.answer_source) for c in result.cases] == [
        (f"{RUN}:000000", "documents"),
        (f"{RUN}:000001", "documents"),
    ]
    # Case 1: the fan tray is a plausible cause, so its shared remedy is a hard negative there.
    graded = _graded(_links(result, 1), chapter)
    assert graded[_section_titled(chapter, "Failed fan tray")] == ("hard_negative", 0)
    assert graded[_section_titled(chapter, "Dust in the heat sink")] == ("resolves", 2)
    assert "sec-5.1" not in graded  # its symptoms show no error code


def test_the_chapter_has_one_remedy_section_per_distinct_cause(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    chapter = _docs(_build(_ctx(company, registry)))["service_manual:"]
    remedies = chapter.section("sec-5.3").subsections
    titles = [s.title for s in remedies]
    assert sorted(titles) == sorted(
        ["Failed fan tray", "Blocked air inlet", "Faulty power supply", "Dust in the heat sink"]
    )
    assert len(chapter.section("sec-5.2").subsections) == 2


def test_a_partial_cause_is_listed_without_its_fix(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    coverage: dict[str, dict[str, Coverage]] = {
        P0: {"Failed fan tray": "partial"},
        P1: {"Failed fan tray": "partial"},
    }
    result = _build(_ctx(company, registry, coverage))
    docs = _docs(result)
    kb0, chapter = docs["kb_article:" + P0], docs["service_manual:"]
    assert "| Failed fan tray | - |" in _text(kb0)  # listed, no remedy to point at
    for doc in (kb0, chapter):
        assert "Watch the fan status LED" not in _text(doc)  # its remedy's own steps
    with pytest.raises(LookupError):
        _section_titled(kb0, "Failed fan tray")
    links = _links(result, 0)
    assert not [link for link in links if link.grade == 2]
    assert not [link for link in links if link.relation == "undocumented"]
    assert ("supports", 1) in _graded(links, kb0).values()
    assert result.cases[0].answer_source == "partial"


def test_an_agent_only_cause_is_in_no_document_and_its_case_is_undocumented(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    coverage: dict[str, dict[str, Coverage]] = {
        P0: {"Failed fan tray": "agent_only"},
        P1: {"Failed fan tray": "agent_only"},
    }
    result = _build(_ctx(company, registry, coverage))
    for built in result.documents:
        assert "fan tray, front" not in _text(built.document).lower()  # its part is gone too
        assert "failed fan tray" not in _text(built.document).lower()
    links = _links(result, 0)
    assert not [link for link in links if link.grade == 2]
    undocumented = [link for link in links if link.relation == "undocumented"]
    assert undocumented == [
        DocumentLink(
            case_uid=f"{RUN}:000000",
            doc_id=None,
            relation="undocumented",
            grade=0,
            basis="root_cause_fix",
            cause="Failed fan tray",
        )
    ]
    # The checks still help: supporting links remain.
    assert [link.relation for link in links if link.grade == 1]
    assert result.cases[0].answer_source == "agent_knowledge"
    assert result.cases[1].answer_source == "documents"


def test_a_problem_with_nothing_documented_gets_no_article(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    coverage: dict[str, dict[str, Coverage]] = {
        P1: {"Failed fan tray": "agent_only", "Dust in the heat sink": "agent_only"},
        P0: {"Failed fan tray": "agent_only"},
    }
    result = _build(_ctx(company, registry, coverage))
    assert sorted(_docs(result)) == ["kb_article:" + P0, "service_manual:"]
    assert [link.relation for link in _links(result, 1)] == ["undocumented"]


def test_building_is_deterministic(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    assert _build(_ctx(company, registry)) == _build(_ctx(company, registry))


def test_problems_without_a_plan_or_a_known_model(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    ctx = _ctx(company, registry)
    ctx.problems[1].diagnosis_plan = None
    ctx.problem_models = {}
    result = _build(ctx)
    docs = _docs(result)
    assert sorted(docs) == ["kb_article:" + P0]  # no model: no chapter
    assert docs["kb_article:" + P0].applies_to.models == []
    assert [c.case_uid for c in result.cases] == [f"{RUN}:000000"]
    assert not _links(result, 1)


def test_an_identifier_the_registry_lacks_fails_the_build(
    company: CompanyProfile, registry: dict[str, ProductFacts]
) -> None:
    ctx = _ctx(company, registry)
    plan = DiagnosisPlan.model_validate(ctx.problems[0].diagnosis_plan)
    plan.candidate_causes[1].resolution_steps = ["Clear code F-99 from the log"]
    ctx.problems[0].diagnosis_plan = plan.model_dump()
    with pytest.raises(ValueError, match="F-99"):
        _build(ctx)


def test_kb_numbers_are_seeded_unique_and_dated_before_the_first_contact() -> None:
    ids = [f"r:p:{i:04d}" for i in range(30)]
    refs = kb_documents("Acme Cloud", ids, seed=3, calendar_start=START)
    assert refs == kb_documents("Acme Cloud", list(reversed(ids)), seed=3, calendar_start=START)
    assert refs != kb_documents("Acme Cloud", ids, seed=4, calendar_start=START)
    numbers = [r.doc_number for r in refs.values()]
    assert len(set(numbers)) == len(numbers)
    assert all(n.startswith("AC-KB-") for n in numbers)
    for r in refs.values():
        assert r.issued == sorted(r.issued) and r.issued[-1] < START.date()
        assert len(r.revisions) == len(r.issued)


def _section_titled(doc: DocumentIR, title: str) -> str:
    """The id of the remedy section titled ``title``."""
    for s, _, parents in doc.walk():
        if s.title == title and parents and parents[-1].title == "Remedies":
            return s.id
        if s.title == title and parents and parents[-1].title.startswith("Symptoms"):
            return s.id
    raise LookupError(title)
