"""Build documents for a finished run (``csfd documents <run_id>``).

Reads what the run committed (its problems, including problems reused with
``--problems-from``, and its cases with their seeded facts and plan), runs the
configured builders over it, checks their output and stores it, replacing any
earlier build of the same run. No model is called here; builders that need one
bring their own.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from csfd.documents.build import (
    BuildContext,
    BuildResult,
    CaseInput,
    DocumentBuilder,
    run_builders,
)
from csfd.facts import CaseFacts
from csfd.seeds.company import CompanyProfile, parse_company_seed
from csfd.storage.db import Database
from csfd.storage.documents import DocumentRepo
from csfd.storage.repository import LineageRepo, ProblemRepo, RunRepo


def load_context(db: Database, run_id: str, *, seeds_dir: str | Path = "seeds") -> BuildContext:
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
    return BuildContext(
        run_id=run_id,
        run_seed=run.run_seed,
        company_name=company_info.get("name") or (company.name if company else "the company"),
        company=company,
        calendar_start=datetime.fromisoformat(calendar) if calendar else None,
        problems=ProblemRepo(db).list_in_run_scope(run_id),
        cases=cases,
    )


def build_run_documents(
    db: Database,
    run_id: str,
    builders: Sequence[DocumentBuilder],
    *,
    seeds_dir: str | Path = "seeds",
) -> BuildResult:
    """Run ``builders`` over a finished run and store the result (replacing an earlier build)."""
    ctx = load_context(db, run_id, seeds_dir=seeds_dir)
    result, made_by = run_builders(ctx, builders)
    DocumentRepo(db).replace_for_run(run_id, result, made_by)
    return result
