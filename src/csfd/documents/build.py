"""The builder interface: what a document builder gets and what it returns.

A builder turns a finished run (company, problems with their diagnosis plans,
cases with their seeded facts) into documents, the images they show, and the
ground-truth links "case -> document section -> relevance grade" that retrieval
evaluation scores against. Builders are registered by name in :data:`BUILDERS`
and selected with ``documents.builders``; none ship yet, so a run gets
documents only once a builder is registered.

Documentation is allowed to be incomplete on purpose: some fixes are known only
to the support agent, never written down. ``BuildContext.coverage`` says, per
problem and candidate cause, whether a builder may document it, and
``BuildContext.answer_sources`` says per case where its answer lives; both are
inputs a coverage planner fills in, empty meaning "documented".
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from csfd.documents.ir import Asset, DocumentIR
from csfd.facts import CaseFacts
from csfd.seeds.company import CompanyProfile
from csfd.storage.repository import ProblemRecord

# Per candidate cause: in the documents, documented without its fix, or agent knowledge only.
Coverage = Literal["documented", "partial", "agent_only"]
# Per case: where the case's fix can be found.
AnswerSource = Literal["documents", "partial", "agent_knowledge"]
Relation = Literal["resolves", "supports", "equivalent", "hard_negative"]

# The grade each relation carries; "equivalent" takes the grade of the section it matches.
_RELATION_GRADES: dict[str, frozenset[int]] = {
    "resolves": frozenset({2}),
    "supports": frozenset({1}),
    "equivalent": frozenset({1, 2}),
    "hard_negative": frozenset({0}),
}


class DocumentLink(BaseModel):
    """Ground truth: how relevant one document section is to one case (or one contact)."""

    case_uid: str  # lineage.ticket_uid
    round_index: int | None = None  # None = the whole case
    doc_id: str
    section_id: str | None = None  # None = the whole document (e.g. a requested manual)
    relation: Relation
    grade: int  # 2 resolves, 1 supports, 0 hard negative
    basis: str  # why, e.g. "root_cause_fix", "check:2", "superseded_revision"

    @model_validator(mode="after")
    def _grade_fits_relation(self) -> DocumentLink:
        if self.grade not in _RELATION_GRADES[self.relation]:
            raise ValueError(f"relation {self.relation!r} cannot have grade {self.grade}")
        return self


class CaseAnswer(BaseModel):
    case_uid: str
    answer_source: AnswerSource


class CaseInput(BaseModel):
    """One case of the run, as builders see it."""

    case_uid: str
    slot_index: int
    problem_id: str
    ticket_type: str
    customer_tier: str
    facts: CaseFacts | None = None
    planned_problem_state: str | None = None
    case_plan: dict[str, Any] | None = None


class BuildContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    run_seed: int
    company_name: str
    company: CompanyProfile | None = None  # the parsed seed, when its file is available
    # Documents are dated before the first contact (tickets.calendar.start).
    calendar_start: datetime | None = None
    problems: list[ProblemRecord] = Field(default_factory=list)
    cases: list[CaseInput] = Field(default_factory=list)
    # problem id -> candidate cause -> coverage; a missing entry means "documented".
    coverage: dict[str, dict[str, Coverage]] = Field(default_factory=dict)
    # case uid -> where its answer lives; a missing entry means not decided.
    answer_sources: dict[str, AnswerSource] = Field(default_factory=dict)

    def coverage_of(self, problem_id: str, cause: str) -> Coverage:
        return self.coverage.get(problem_id, {}).get(cause, "documented")


class BuiltDocument(BaseModel):
    document: DocumentIR
    assets: list[Asset] = Field(default_factory=list)


class BuildResult(BaseModel):
    documents: list[BuiltDocument] = Field(default_factory=list)
    links: list[DocumentLink] = Field(default_factory=list)
    cases: list[CaseAnswer] = Field(default_factory=list)


class DocumentBuilder(Protocol):
    """Makes documents for a run. Must be deterministic for a given context."""

    name: str

    def build(self, ctx: BuildContext) -> BuildResult: ...


# Builder name -> factory. Later work registers its builders here.
BUILDERS: dict[str, Callable[[], DocumentBuilder]] = {}


def resolve_builders(
    names: Sequence[str], registry: Mapping[str, Callable[[], DocumentBuilder]] | None = None
) -> list[DocumentBuilder]:
    """Instantiate the named builders; an unknown name is an error naming the known ones."""
    reg = BUILDERS if registry is None else registry
    unknown = [n for n in names if n not in reg]
    if unknown:
        known = ", ".join(sorted(reg)) or "none registered"
        raise ValueError(f"unknown document builder(s): {', '.join(unknown)} (known: {known})")
    return [reg[n]() for n in names]


def run_builders(
    ctx: BuildContext, builders: Sequence[DocumentBuilder]
) -> tuple[BuildResult, dict[str, str]]:
    """Run every builder, merge and check their output.

    Returns the merged result and which builder made each document. Fails on a
    duplicate document id or export folder, a link to an unknown document,
    section or case, a figure without its image, or a document that documents
    a cause whose coverage is ``agent_only`` (see :func:`_agent_only_causes`).
    """
    merged = BuildResult()
    made_by: dict[str, str] = {}
    for builder in builders:
        result = builder.build(ctx)
        for built in result.documents:
            doc_id = built.document.doc_id
            if doc_id in made_by:
                raise ValueError(
                    f"document {doc_id} built twice ({made_by[doc_id]}, {builder.name})"
                )
            made_by[doc_id] = builder.name
        merged.documents.extend(result.documents)
        merged.links.extend(result.links)
        merged.cases.extend(result.cases)
    _check(ctx, merged)
    return merged, made_by


def _cause_key(cause: str) -> str:
    """Cause text compared loosely: lower case, punctuation and extra spaces removed."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", cause.lower()).split())


def _agent_only_causes(ctx: BuildContext, doc: DocumentIR) -> set[str]:
    """Causes the document must not document: agent knowledge for the problems it serves.

    A problem-tier document serves its problem and a case-tier document its
    case's problem. A library document is shared by every case, so it must not
    document a cause that is agent knowledge for any problem of the run.
    """
    if doc.tier == "problem":
        problems = [doc.problem_id]
    elif doc.tier == "case":
        problems = [c.problem_id for c in ctx.cases if c.case_uid == doc.case_uid]
    else:
        problems = list(ctx.coverage)
    return {
        _cause_key(cause)
        for pid in problems
        if pid is not None
        for cause, level in ctx.coverage.get(pid, {}).items()
        if level == "agent_only"
    }


def _check(ctx: BuildContext, result: BuildResult) -> None:
    docs = {b.document.doc_id: b for b in result.documents}
    problem_ids = {p.id for p in ctx.problems}
    case_uids = {c.case_uid for c in ctx.cases}
    folders: dict[str, str] = {}
    for built in result.documents:
        doc = built.document
        folder = doc.folder_name
        if folder in folders:
            raise ValueError(f"documents {folders[folder]} and {doc.doc_id} share {folder!r}")
        folders[folder] = doc.doc_id
        if doc.tier == "problem" and doc.problem_id not in problem_ids:
            raise ValueError(f"document {doc.doc_id} serves unknown problem {doc.problem_id}")
        if doc.tier == "case" and doc.case_uid not in case_uids:
            raise ValueError(f"document {doc.doc_id} serves unknown case {doc.case_uid}")
        have = {a.asset_id for a in built.assets}
        for fig in doc.figures():
            if fig.asset_id not in have:
                raise ValueError(f"document {doc.doc_id}: figure {fig.id} has no asset")
        hidden = _agent_only_causes(ctx, doc)
        for s, _, _ in doc.walk():
            for cause in s.covers:
                if _cause_key(cause) in hidden:
                    raise ValueError(
                        f"document {doc.doc_id} section {s.id} documents {cause!r}, "
                        "which is agent knowledge only"
                    )
    for link in result.links:
        if link.case_uid not in case_uids:
            raise ValueError(f"link to unknown case {link.case_uid}")
        built_doc = docs.get(link.doc_id)
        if built_doc is None:
            raise ValueError(f"link to unknown document {link.doc_id}")
        if link.section_id is not None:
            try:
                built_doc.document.section(link.section_id)
            except KeyError as exc:
                raise ValueError(f"link to unknown section: {exc.args[0]}") from exc
    answered = [c.case_uid for c in result.cases]
    if len(set(answered)) != len(answered):
        raise ValueError("a case has more than one answer source")
    for uid in answered:
        if uid not in case_uids:
            raise ValueError(f"answer source for unknown case {uid}")
