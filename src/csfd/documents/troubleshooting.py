"""Troubleshooting documents built from the problems' diagnosis plans (no model call).

Two kinds of document, both written only from what Phase 1 committed:

* a **knowledge-base article per problem** (problem tier, internal): summary,
  symptoms (with the registry error codes they show), safety notes, a
  diagnosis section (possible causes and the ordered checks) and one remedy
  section per documented cause;
* a **troubleshooting chapter per machine model** (chapter 5 of the model's
  service manual, library tier): the model's error codes, one symptom section
  per problem on that model, and one shared remedy section per distinct cause,
  as real manuals do. Its number, revision and history are the registry's
  service-manual entry, so the full manual can later grow around it with the
  same document id and section ids.

Causes appear in the guide's neutral, seeded order (:func:`csfd.diagnosis.guide_causes`),
and nothing marks the root cause or states a check's result. Coverage decides what
is written: a ``partial`` cause is listed with its checks but has no remedy
section; an ``agent_only`` cause is not mentioned at all.

Ground truth comes by construction, per case: the root cause's remedy sections
resolve it (grade 2); the diagnosis and symptom sections with its checks, and
the error-code sections for codes its symptoms show, support it (grade 1); the
other causes' remedy sections are hard negatives (grade 0); another problem's
article with a remedy for the same cause on the same model is equivalent
(grade 2). A root cause that is agent knowledge only gets an ``undocumented``
link instead. Every number is the registry's: document numbers, part numbers,
error codes, and cross-references resolved from placeholders
(:mod:`csfd.documents.placeholders`); a number the problems' text names but the
registry lacks is left out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from csfd.diagnosis import CandidateCause, DiagnosisPlan, guide_causes
from csfd.documents.build import (
    BuildContext,
    BuildResult,
    BuiltDocument,
    CaseAnswer,
    DocumentLink,
)
from csfd.documents.coverage import answer_source, cause_key
from csfd.documents.ir import (
    Admonition,
    AppliesTo,
    Block,
    DocType,
    DocumentIR,
    ItemList,
    Paragraph,
    Procedure,
    Revision,
    Section,
    Step,
    Table,
    make_doc_id,
)
from csfd.documents.placeholders import map_text, resolve_document
from csfd.documents.registry import (
    DocRef,
    ErrorCode,
    ProductFacts,
    cited_error_codes,
    find_part,
    kb_documents,
    part_name,
    strip_unknown_identifiers,
    text_values,
)
from csfd.storage.repository import ProblemRecord

# The service manual's troubleshooting chapter (report skeleton: 5 Troubleshooting).
CHAPTER = 5
AUTHOR = "Technical Documentation"
APPROVER = "Technical Service"
_TITLES: dict[DocType, str] = {
    "operating_manual": "Operating manual",
    "service_manual": "Service manual",
    "datasheet": "Technical datasheet",
    "parts_list": "Spare parts list",
}


@dataclass
class _Problem:
    """One problem's documented content."""

    record: ProblemRecord
    plan: DiagnosisPlan
    model: str | None
    listed: list[CandidateCause]  # documented or partial, in the guide's order
    documented: list[CandidateCause]  # with a remedy section
    codes: list[ErrorCode] = field(default_factory=list)  # error codes its symptoms show
    hides_fix: bool = False  # a cause's fix is not documented (partial or agent only)


@dataclass
class _Built:
    doc: DocumentIR
    diagnosis: str | None = None  # section with the possible causes and the checks
    codes: str | None = None  # section with an error-code table the case's symptoms show
    remedies: dict[str, str] = field(default_factory=dict)  # cause key -> remedy section id
    symptoms: dict[str, str] = field(default_factory=dict)  # problem id -> symptom section id
    problem_codes: dict[str, bool] = field(default_factory=dict)  # problem id -> shows a code


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


def _revisions(ref: DocRef, first: str, later: str) -> list[Revision]:
    return [
        Revision(rev=rev, issued=issued, description=first if i == 0 else later, author=AUTHOR)
        for i, (rev, issued) in enumerate(zip(ref.revisions, ref.issued, strict=True))
    ]


def _parts_table(
    table_id: str, caption: str, cause: CandidateCause, ctx: BuildContext, model: str | None
) -> Table | None:
    rows: list[list[str]] = []
    for entry in cause.parts:
        part = find_part(entry, ctx.registry, model)
        if part is not None:
            rows.append([part.part_number, part.name, str(part.item) if part.item else "-"])
        else:  # a part the registry does not number: named, never given an invented number
            rows.append(["-", part_name(entry, ctx.registry) or "-", "-"])
    if not rows:
        return None
    return Table(
        id=table_id,
        caption=caption,
        columns=["Part number", "Description", "Item"],
        rows=rows,
        role="parts",
    )


def _remedy_blocks(
    prefix: str, number: str, cause: CandidateCause, ctx: BuildContext, model: str | None
) -> list[Block]:
    blocks: list[Block] = [
        Paragraph(id=f"{prefix}-cause", text=f"Cause: {_sentence(cause.cause)}."),
    ]
    steps = [Step(id="", text=_sentence(s)) for s in cause.resolution_steps if s.strip()]
    if cause.verification:
        # The last step checks the fix; its expected result is what the customer then sees.
        steps.append(
            Step(
                id="",
                text=f"Check the result: {_sentence(cause.verification)}",
                expected=_sentence(cause.verification_finding) or None,
            )
        )
    if steps:
        blocks.append(
            Procedure(
                id=f"{prefix}-proc",
                title="Remedy",
                steps=[
                    st.model_copy(update={"id": f"{prefix}-proc-s{i}"})
                    for i, st in enumerate(steps, 1)
                ],
            )
        )
    parts = _parts_table(f"{prefix}-parts", f"Table {number}-1. Spare parts", cause, ctx, model)
    if parts is not None:
        blocks.append(parts)
    if cause.workaround:
        blocks.append(
            Admonition(
                id=f"{prefix}-workaround",
                level="note",
                text=f"Until the remedy is applied: {_sentence(cause.workaround)}",
            )
        )
    if cause.preventive_action:
        blocks.append(
            Paragraph(
                id=f"{prefix}-prevent", text=f"Prevention: {_sentence(cause.preventive_action)}"
            )
        )
    return blocks


def _checks_procedure(proc_id: str, plan: DiagnosisPlan) -> Procedure | None:
    """The ordered checks, without their results or what each rules out."""
    if not plan.checks:
        return None
    return Procedure(
        id=proc_id,
        title="Diagnostic checks, in this order",
        steps=[
            Step(id=f"{proc_id}-s{i}", text=f"{_sentence(c.check)}: {_sentence(c.how_to_check)}")
            for i, c in enumerate(plan.checks, 1)
        ],
    )


def _causes_table(table_id: str, caption: str, p: _Problem, remedy_ids: dict[str, str]) -> Table:
    rows = [
        [
            _sentence(c.cause),
            f"See {{section:{remedy_ids[cause_key(c.cause)]}}}"
            if cause_key(c.cause) in remedy_ids
            else "-",
        ]
        for c in p.listed
    ]
    return Table(
        id=table_id,
        caption=caption,
        columns=["Possible cause", "Remedy"],
        rows=rows,
    )


def _codes_table(
    table_id: str,
    caption: str,
    model: str,
    codes: Sequence[ErrorCode],
    withheld: set[tuple[str, str]],
) -> Table:
    """Codes with their meaning and first action; a ``withheld`` code's action is left out."""
    return Table(
        id=table_id,
        caption=caption,
        columns=["Code", "Meaning", "Action"],
        rows=[
            [c.code, c.meaning, "-" if (model, c.code) in withheld else c.action or "-"]
            for c in codes
        ],
        role="error_codes",
    )


def _withheld_actions(problems: Sequence[_Problem], ctx: BuildContext) -> set[tuple[str, str]]:
    """The ``(model, code)`` pairs whose registry action must not be printed.

    A registry action can state the fix of a cause whose fix is not documented
    (an error code's "replace the fan tray" is the fix of "failed fan tray"),
    so every code a problem with a partial or agent-only cause cites, anywhere
    in its text or plan, is printed without its action. A chapter lists every
    code of its model, cited or not, so any code whose action states such a
    cause's fix text is withheld too.
    """
    withheld: set[tuple[str, str]] = set()
    hidden: set[str] = set()
    for p in problems:
        if not p.hides_fix:
            continue
        r = p.record
        texts = [r.title, r.summary, r.background, *r.symptoms, *text_values(p.plan.model_dump())]
        models = [p.model] if p.model in ctx.registry else list(ctx.registry)
        for model in models:
            assert model is not None
            withheld |= {(model, c.code) for c in cited_error_codes(texts, ctx.registry[model])}
        for cause in p.plan.candidate_causes:
            if ctx.coverage_of(r.id, cause.cause) != "documented":
                fixes = [*cause.resolution_steps, cause.verification, cause.workaround]
                hidden |= {cause_key(t) for t in fixes if t and cause_key(t)}
    for model, facts in ctx.registry.items():
        for c in facts.error_codes:
            action = cause_key(c.action or "")
            if action and any(h in action for h in hidden):
                withheld.add((model, c.code))
    return withheld


def _related(facts: ProductFacts | None) -> list[str]:
    if facts is None:
        return []
    return [
        f"{_TITLES[d.doc_type]}: {{doc:{d.doc_type}}}"
        for d in facts.documents
        if d.doc_type in _TITLES
    ]


def _kb_article(
    p: _Problem, ref: DocRef, ctx: BuildContext, withheld: set[tuple[str, str]]
) -> _Built:
    record, plan = p.record, p.plan
    facts = ctx.registry.get(p.model) if p.model else None
    built_sections: list[Section] = []
    codes_section: str | None = None

    def number() -> str:
        return str(len(built_sections) + 1)

    n = number()
    summary: list[Block] = [Paragraph(id="p-summary", text=_sentence(record.summary))]
    if p.model:
        summary.append(Paragraph(id="p-applies", text=f"Applies to: {p.model}."))
    built_sections.append(Section(id=f"sec-{n}", number=n, title="Summary", blocks=summary))

    n = number()
    symptoms: list[Block] = []
    if record.symptoms:
        symptoms.append(ItemList(id="l-symptoms", items=[_sentence(s) for s in record.symptoms]))
    else:
        symptoms.append(Paragraph(id="p-symptoms", text=_sentence(record.title)))
    if p.codes:
        symptoms.append(
            _codes_table(
                "t-codes",
                f"Table {n}-1. Codes shown with this symptom",
                p.model or "",
                p.codes,
                withheld,
            )
        )
        codes_section = f"sec-{n}"
    built_sections.append(Section(id=f"sec-{n}", number=n, title="Symptoms", blocks=symptoms))

    if plan.safety:
        n = number()
        built_sections.append(
            Section(
                id=f"sec-{n}",
                number=n,
                title="Safety",
                blocks=[
                    Admonition(id=f"a-safety-{i}", level="warning", text=_sentence(note))
                    for i, note in enumerate(plan.safety, 1)
                ],
            )
        )

    diagnosis_number = number()
    remedies_number = str(int(diagnosis_number) + 1)
    remedy_ids = {
        cause_key(c.cause): f"sec-{remedies_number}.{i}" for i, c in enumerate(p.documented, 1)
    }
    diagnosis: list[Block] = [
        Paragraph(
            id="p-diagnosis",
            text="Run the checks below with the customer; their results show which cause "
            "applies. Then apply the remedy for that cause.",
        ),
        _causes_table("t-causes", f"Table {diagnosis_number}-1. Possible causes", p, remedy_ids),
    ]
    checks = _checks_procedure("proc-checks", plan)
    if checks is not None:
        diagnosis.append(checks)
    built_sections.append(
        Section(
            id=f"sec-{diagnosis_number}",
            number=diagnosis_number,
            title="Diagnosis",
            blocks=diagnosis,
            covers=[c.cause for c in p.listed],
        )
    )

    if p.documented:
        subsections = []
        for i, c in enumerate(p.documented, 1):
            num = f"{remedies_number}.{i}"
            subsections.append(
                Section(
                    id=f"sec-{num}",
                    number=num,
                    title=_sentence(c.cause),
                    blocks=_remedy_blocks(f"r{i}", num, c, ctx, p.model),
                    covers=[c.cause],
                )
            )
        built_sections.append(
            Section(
                id=f"sec-{remedies_number}",
                number=remedies_number,
                title="Remedies",
                subsections=subsections,
            )
        )

    related = _related(facts)
    if related:
        n = number()
        built_sections.append(
            Section(
                id=f"sec-{n}",
                number=n,
                title="Related documents",
                blocks=[ItemList(id="l-related", items=related)],
            )
        )

    doc = DocumentIR(
        doc_id=make_doc_id(ctx.company_name, ref.doc_number, "en", ref.revisions[-1]),
        doc_type="kb_article",
        tier="problem",
        doc_number=ref.doc_number,
        title=_sentence(record.title),
        subtitle=f"Knowledge base article. Approved by {APPROVER}.",
        company=ctx.company_name,
        revision=ref.revisions[-1],
        issue_date=ref.issued[-1],
        audience="internal",
        applies_to=AppliesTo(models=[p.model] if p.model else []),
        revisions=_revisions(ref, "First publication", "Causes and checks reviewed"),
        sections=built_sections,
        problem_id=record.id,
    )
    return _Built(
        doc=resolve_document(doc, facts),
        diagnosis=f"sec-{diagnosis_number}",
        codes=codes_section,
        remedies=remedy_ids,
        problem_codes={record.id: bool(p.codes)},
    )


def _chapter(
    model: str, problems: list[_Problem], ctx: BuildContext, withheld: set[tuple[str, str]]
) -> _Built | None:
    facts = ctx.registry[model]
    ref = next((d for d in facts.documents if d.doc_type == "service_manual"), None)
    if ref is None:
        return None
    top = str(CHAPTER)
    codes_section: str | None = None
    symptom_ids: dict[str, str] = {}
    subsections: list[Section] = []

    def number() -> str:
        return f"{top}.{len(subsections) + 1}"

    if facts.error_codes:
        n = number()
        codes_section = f"sec-{n}"
        subsections.append(
            Section(
                id=codes_section,
                number=n,
                title="Error and warning codes",
                blocks=[
                    _codes_table(
                        "t-codes",
                        f"Table {top}-1. Error and warning codes",
                        model,
                        facts.error_codes,
                        withheld,
                    )
                ],
            )
        )

    symptoms_number = number()
    remedies_number = f"{top}.{len(subsections) + 2}"
    # One remedy section per distinct cause, in order of first appearance.
    remedy_causes: dict[str, CandidateCause] = {}
    for p in problems:
        for c in p.documented:
            remedy_causes.setdefault(cause_key(c.cause), c)
    remedy_ids = {key: f"sec-{remedies_number}.{i}" for i, key in enumerate(remedy_causes, 1)}

    symptom_sections = []
    for i, p in enumerate(problems, 1):
        num = f"{symptoms_number}.{i}"
        sid = f"sec-{num}"
        symptom_ids[p.record.id] = sid
        blocks: list[Block] = []
        if p.record.symptoms:
            blocks.append(
                ItemList(id=f"s{i}-symptoms", items=[_sentence(s) for s in p.record.symptoms])
            )
        blocks += [
            Admonition(id=f"s{i}-safety-{j}", level="warning", text=_sentence(note))
            for j, note in enumerate(p.plan.safety, 1)
        ]
        blocks.append(
            _causes_table(f"s{i}-causes", f"Table {num}-1. Possible causes", p, remedy_ids)
        )
        checks = _checks_procedure(f"s{i}-checks", p.plan)
        if checks is not None:
            blocks.append(checks)
        symptom_sections.append(
            Section(
                id=sid,
                number=num,
                title=_sentence(p.record.title),
                blocks=blocks,
                covers=[c.cause for c in p.listed],
            )
        )
    subsections.append(
        Section(
            id=f"sec-{symptoms_number}",
            number=symptoms_number,
            title="Symptoms, causes and checks",
            blocks=[
                Paragraph(
                    id="p-symptoms",
                    text="Find the symptom, run its checks in order, then apply the remedy "
                    "for the cause the checks point to.",
                )
            ],
            subsections=symptom_sections,
        )
    )

    if remedy_causes:
        remedy_sections = []
        for i, (key, c) in enumerate(remedy_causes.items(), 1):
            num = f"{remedies_number}.{i}"
            remedy_sections.append(
                Section(
                    id=remedy_ids[key],
                    number=num,
                    title=_sentence(c.cause),
                    blocks=_remedy_blocks(f"r{i}", num, c, ctx, model),
                    covers=[c.cause],
                )
            )
        subsections.append(
            Section(
                id=f"sec-{remedies_number}",
                number=remedies_number,
                title="Remedies",
                subsections=remedy_sections,
            )
        )

    description = next(
        (
            a.description
            for a in (ctx.company.case_facts.assets if ctx.company else [])
            if a.model == model and a.description
        ),
        None,
    )
    doc = DocumentIR(
        doc_id=make_doc_id(ctx.company_name, ref.doc_number, "en", ref.revisions[-1]),
        doc_type="service_manual",
        tier="library",
        doc_number=ref.doc_number,
        title=f"{model} {description}" if description else f"{model} service manual",
        subtitle=f"Service manual, chapter {CHAPTER}: Troubleshooting",
        company=ctx.company_name,
        revision=ref.revisions[-1],
        issue_date=ref.issued[-1],
        audience="internal",
        applies_to=AppliesTo(models=[model]),
        revisions=_revisions(ref, "First release", "Troubleshooting updated"),
        sections=[
            Section(
                id=f"sec-{top}",
                number=top,
                title="Troubleshooting",
                blocks=[
                    Admonition(
                        id="a-service",
                        level="warning",
                        text="Only trained service staff may carry out the remedies in this "
                        "chapter. Disconnect the unit from mains power before opening it.",
                    )
                ],
                subsections=subsections,
            )
        ],
    )
    return _Built(
        doc=resolve_document(doc, facts),
        codes=codes_section,
        remedies=remedy_ids,
        symptoms=symptom_ids,
        problem_codes={p.record.id: bool(p.codes) for p in problems},
    )


def _drop_unknown_identifiers(doc: DocumentIR, ctx: BuildContext, model: str | None) -> DocumentIR:
    """``doc`` without the part numbers and error codes the registry lacks.

    Problems of a run generated with documents disabled were written without
    the registry, so their text can name numbers it does not hold.
    """
    if ctx.company is None or not ctx.registry:
        return doc
    catalogue = ctx.company.case_facts
    return map_text(
        doc, lambda text: strip_unknown_identifiers(text, ctx.registry, model, catalogue)
    )


class TroubleshootingBuilder:
    """KB articles per problem and a troubleshooting chapter per machine model."""

    name = "troubleshooting"

    def build(self, ctx: BuildContext) -> BuildResult:
        problems = self._problems(ctx)
        result = BuildResult()
        if not problems:
            return result
        if ctx.calendar_start is None:
            raise ValueError("troubleshooting documents need the run's calendar start")
        refs = kb_documents(
            ctx.company_name,
            [p.record.id for p in problems if p.listed],
            seed=ctx.run_seed,
            calendar_start=ctx.calendar_start,
        )
        withheld = _withheld_actions(problems, ctx)
        articles: dict[str, _Built] = {
            p.record.id: _kb_article(p, refs[p.record.id], ctx, withheld)
            for p in problems
            if p.listed
        }
        chapters: dict[str, _Built] = {}
        for model in ctx.registry:
            on_model = [p for p in problems if p.model == model and p.listed]
            chapter = _chapter(model, on_model, ctx, withheld) if on_model else None
            if chapter is not None:
                chapters[model] = chapter
        models = {p.record.id: p.model for p in problems}
        for pid, built in articles.items():
            built.doc = _drop_unknown_identifiers(built.doc, ctx, models[pid])
        for model, built in chapters.items():
            built.doc = _drop_unknown_identifiers(built.doc, ctx, model)
        result.documents = [BuiltDocument(document=b.doc) for b in articles.values()] + [
            BuiltDocument(document=b.doc) for b in chapters.values()
        ]
        by_id = {p.record.id: p for p in problems}
        for case in ctx.cases:
            p = by_id.get(case.problem_id)
            root = p.plan.root_cause() if p else None
            if p is None or root is None:
                continue
            result.cases.append(
                CaseAnswer(
                    case_uid=case.case_uid,
                    answer_source=answer_source(ctx.coverage_of(p.record.id, root.cause)),
                )
            )
            result.links += _links(case.case_uid, p, root, ctx, articles, chapters)
        return result

    def _problems(self, ctx: BuildContext) -> list[_Problem]:
        out = []
        for record in ctx.problems:
            if not record.diagnosis_plan:
                continue
            plan = DiagnosisPlan.model_validate(record.diagnosis_plan)
            if not plan.candidate_causes:
                continue
            ordered = guide_causes(plan, seed_label=f"{ctx.run_seed}:{record.id}")
            levels = [(c, ctx.coverage_of(record.id, c.cause)) for c in ordered]
            model = ctx.problem_models.get(record.id)
            facts = ctx.registry.get(model) if model else None
            texts = [record.title, record.summary, *record.symptoms]
            out.append(
                _Problem(
                    record=record,
                    plan=plan,
                    model=model,
                    listed=_unique([c for c, lv in levels if lv != "agent_only"]),
                    documented=_unique([c for c, lv in levels if lv == "documented"]),
                    codes=cited_error_codes(texts, facts) if facts else [],
                    hides_fix=any(lv != "documented" for _, lv in levels),
                )
            )
        return out


def _unique(causes: list[CandidateCause]) -> list[CandidateCause]:
    """Causes with distinct text (a plan repeating a cause gets it listed once)."""
    seen: dict[str, CandidateCause] = {}
    for c in causes:
        seen.setdefault(cause_key(c.cause), c)
    return list(seen.values())


def _links(
    case_uid: str,
    p: _Problem,
    root: CandidateCause,
    ctx: BuildContext,
    articles: dict[str, _Built],
    chapters: dict[str, _Built],
) -> list[DocumentLink]:
    root_key = cause_key(root.cause)
    pid = p.record.id
    links: list[DocumentLink] = []

    def link(
        doc: DocumentIR, section: str, relation: str, grade: int, basis: str, cause: str | None
    ) -> None:
        links.append(
            DocumentLink.model_validate(
                {
                    "case_uid": case_uid,
                    "doc_id": doc.doc_id,
                    "section_id": section,
                    "relation": relation,
                    "grade": grade,
                    "basis": basis,
                    "cause": cause,
                }
            )
        )

    def remedies(built: _Built) -> None:
        for c in p.documented:
            key = cause_key(c.cause)
            if key not in built.remedies:
                continue
            if key == root_key:
                link(built.doc, built.remedies[key], "resolves", 2, "root_cause_fix", c.cause)
            else:
                link(built.doc, built.remedies[key], "hard_negative", 0, "other_cause", c.cause)

    article = articles.get(pid)
    if article is not None:
        if article.diagnosis:
            link(article.doc, article.diagnosis, "supports", 1, "checks", None)
        if article.codes:
            link(article.doc, article.codes, "supports", 1, "error_codes", None)
        remedies(article)
    chapter = chapters.get(p.model) if p.model else None
    if chapter is not None and pid in chapter.symptoms:
        link(chapter.doc, chapter.symptoms[pid], "supports", 1, "checks", None)
        if chapter.codes and chapter.problem_codes.get(pid):
            link(chapter.doc, chapter.codes, "supports", 1, "error_codes", None)
        remedies(chapter)
    # Another problem's article on the same machine with a remedy for the same cause.
    for other_id, other in articles.items():
        if other_id == pid or p.model is None or other.doc.applies_to.models != [p.model]:
            continue
        if root_key in other.remedies:
            link(other.doc, other.remedies[root_key], "equivalent", 2, "same_cause", root.cause)
    if ctx.coverage_of(pid, root.cause) == "agent_only":
        links.append(
            DocumentLink(
                case_uid=case_uid,
                doc_id=None,
                relation="undocumented",
                grade=0,
                basis="root_cause_fix",
                cause=root.cause,
            )
        )
    return links
