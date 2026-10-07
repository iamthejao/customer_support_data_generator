"""Build documents for a finished run (``csfd documents <run_id>``).

Reads what the run committed (its problems, including problems reused with
``--problems-from``, its cases with their seeded facts and plan, and its
identifier registry), plans documentation coverage
(:mod:`csfd.documents.coverage`), runs the configured builders over it, checks
their output and stores it, replacing any earlier build of the same run. No
model is called here; builders that need one bring their own.

A run generated with documents disabled has no stored registry: it gets the
registry code alone draws from the seed (document numbers, revisions, dates,
and the seed tables' parts and codes), without the model-named entries.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from csfd.diagnosis import DiagnosisPlan
from csfd.documents.build import (
    BUILDERS,
    BuildContext,
    BuildResult,
    CaseInput,
    DocumentBuilder,
    run_builders,
)
from csfd.documents.coverage import (
    Coverage,
    answer_source,
    default_coverage_proportions,
    plan_coverage,
)
from csfd.documents.registry import ProductFacts, build_registry, problem_model, text_values
from csfd.documents.troubleshooting import TroubleshootingBuilder
from csfd.facts import CaseFacts
from csfd.seeds.company import CompanyProfile, parse_company_seed
from csfd.storage.db import Database
from csfd.storage.documents import DocumentRepo
from csfd.storage.product_facts import ProductFactsRepo
from csfd.storage.repository import LineageRepo, ProblemRecord, ProblemRepo, RunRepo

BUILDERS.setdefault(TroubleshootingBuilder.name, TroubleshootingBuilder)


def _problem_models(
    problems: list[ProblemRecord],
    cases: list[CaseInput],
    registry: Mapping[str, ProductFacts],
    company: CompanyProfile | None,
) -> dict[str, str | None]:
    """The machine model of each problem, as Phase 2 reads it.

    The model the problem names or whose registry identifiers it cites; else
    the one model all its cases ran on; else unknown.
    """
    out: dict[str, str | None] = {}
    for problem in problems:
        model = None
        if company is not None and registry:
            model = problem_model(
                problem.free_text(),
                text_values(problem.diagnosis_plan),
                registry,
                company.case_facts,
            )
        if model is None:
            seen = {c.facts.asset_model for c in cases if c.problem_id == problem.id and c.facts}
            model = seen.pop() if len(seen) == 1 else None
        out[problem.id] = model
    return out


def load_context(
    db: Database,
    run_id: str,
    *,
    seeds_dir: str | Path = "seeds",
    coverage_proportions: Mapping[Coverage, float] | None = None,
) -> BuildContext:
    """The builders' view of a run. Raises ``KeyError`` for an unknown run."""
    run = RunRepo(db).get(run_id)
    snapshot = json.loads(run.config_snapshot_json or "{}")
    company_info = snapshot.get("company") or {}
    company: CompanyProfile | None = None
    slug = company_info.get("seed")
    if slug:
        path = Path(seeds_dir) / slug / "company_seed.md"
        if path.exists():
            company = parse_company_seed(path)
    calendar = ((snapshot.get("tickets") or {}).get("calendar") or {}).get("start")
    cases = [
        CaseInput(
            case_uid=row.ticket_uid,
            slot_index=row.slot_index,
            problem_id=row.problem_id,
            ticket_type=row.ticket_type,
            customer_tier=row.customer_tier,
            facts=CaseFacts.model_validate(row.case_facts) if row.case_facts else None,
            planned_problem_state=row.problem_state,
            case_plan=row.case_plan,
        )
        for row in LineageRepo(db).list_for_run(run_id)
    ]
    # Documents are dated before the first contact; a run without a calendar uses its start.
    calendar_start = datetime.fromisoformat(calendar) if calendar else run.started_at
    problems = ProblemRepo(db).list_in_run_scope(run_id)
    registry = ProductFactsRepo(db).list_for_run(run_id)
    if not registry and company is not None:
        registry = build_registry(
            company, seed=run.run_seed, calendar_start=calendar_start, names={}
        )
    plans = [
        (p.id, DiagnosisPlan.model_validate(p.diagnosis_plan)) for p in problems if p.diagnosis_plan
    ]
    coverage = plan_coverage(
        plans,
        coverage_proportions or default_coverage_proportions(),
        seed=run.run_seed,
    )
    roots = {pid: plan.root_cause() for pid, plan in plans}
    answer_sources = {}
    for case in cases:
        root = roots.get(case.problem_id)
        if root is not None:
            answer_sources[case.case_uid] = answer_source(coverage[case.problem_id][root.cause])
    return BuildContext(
        run_id=run_id,
        run_seed=run.run_seed,
        company_name=company_info.get("name") or (company.name if company else "the company"),
        company=company,
        calendar_start=calendar_start,
        problems=problems,
        cases=cases,
        coverage=coverage,
        answer_sources=answer_sources,
        registry=registry,
        problem_models=_problem_models(problems, cases, registry, company),
    )


def build_run_documents(
    db: Database,
    run_id: str,
    builders: Sequence[DocumentBuilder],
    *,
    seeds_dir: str | Path = "seeds",
    coverage_proportions: Mapping[Coverage, float] | None = None,
) -> BuildResult:
    """Run ``builders`` over a finished run and store the result (replacing an earlier build)."""
    ctx = load_context(db, run_id, seeds_dir=seeds_dir, coverage_proportions=coverage_proportions)
    result, made_by = run_builders(ctx, builders)
    DocumentRepo(db).replace_for_run(run_id, result, made_by)
    return result
