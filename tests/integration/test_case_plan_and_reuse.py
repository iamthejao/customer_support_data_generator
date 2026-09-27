"""Integration: a case-plan file pins cases, and a later run reuses a run's problems."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import typer

from csfd.case_plan import CasePlan, CasePlanEntry
from csfd.cli import _load_plan, _reused_problem_ids
from csfd.outcomes import ProblemState
from csfd.pipeline import run_pipeline
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import Scenario, ScenarioCatalogue
from csfd.settings import AppSettings, Channel
from csfd.storage.db import Database
from csfd.storage.exporters import export_run_to_jsonl
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import ProblemRepo, RunRepo
from csfd.ticket_types.definitions import TicketType
from tests.integration.test_pipeline_deterministic import _build_fake_factory, _build_settings

COMPANY = CompanyProfile(name="Acme", raw_markdown="# Acme", sections={})
SCENARIOS = ScenarioCatalogue(
    scenarios=[Scenario(category="auth", title="Login issues", summary="Cannot sign in.")],
    source_path=Path("seeds/kalvora/scenarios_seed.md"),
)


def _run(
    settings: AppSettings,
    db: Database,
    tmp_path: Path,
    *,
    case_plan: CasePlan | None = None,
    problems_from: str | None = None,
) -> str:
    return asyncio.run(
        run_pipeline(
            settings=settings,
            factory=_build_fake_factory(tmp_path),
            db=db,
            company=COMPANY,
            scenarios=SCENARIOS,
            case_plan=case_plan,
            problems_from=problems_from,
        )
    )


def _setup(tmp_path: Path, channel: Channel = "email") -> tuple[AppSettings, Database]:
    settings = _build_settings(tmp_path=tmp_path, total=10, problems=3, channel=channel)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    return settings, db


def test_case_plan_pins_every_dimension_it_names(tmp_path: Path) -> None:
    settings, db = _setup(tmp_path)
    plan = CasePlan(
        cases=[
            CasePlanEntry(
                problem=2,
                ticket_type=TicketType.L2,
                tier="enterprise",
                tone="frustrated",
                end_modes=["follow_up"],
                problem_state=ProblemState.PENDING_VISIT,
            ),
            CasePlanEntry(tone="polite"),
        ]
    )
    run_id = _run(settings, db, tmp_path, case_plan=plan)

    problems = ProblemRepo(db).list_for_run(run_id)
    with db.connect() as conn:
        lineage = conn.execute(
            "SELECT * FROM lineage WHERE run_id = ? ORDER BY slot_index", (run_id,)
        ).fetchall()
        contacts = conn.execute(
            "SELECT case_uid, round_index, round_count FROM resolutions WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    # The plan's length is the number of cases, whatever tickets.total says.
    assert len(lineage) == 2
    first, second = lineage
    assert first["problem_id"] == problems[2].id
    assert (first["ticket_type"], first["customer_tier"], first["customer_tone"]) == (
        "l2",
        "enterprise",
        "frustrated",
    )
    assert first["problem_state"] == "pending_visit"
    case_plan = json.loads(first["case_plan_json"])
    assert [c["end_mode"] for c in case_plan["contacts"]] == ["follow_up", "final"]
    assert sorted((c["round_index"], c["round_count"]) for c in contacts)[:2] == [(1, 1), (1, 2)]
    # Unpinned dimensions still come from the proportions; the pinned one holds.
    assert second["customer_tone"] == "polite"
    assert second["ticket_type"] == "l1"  # 0.5/0.3/0.1/0.1 over 2 cases: one docs, one l1
    snapshot = json.loads(RunRepo(db).get(run_id).config_snapshot_json)
    assert snapshot["case_plan"]["cases"][1] == {"tone": "polite"}


def test_problems_from_reuses_problems_in_another_channel(tmp_path: Path) -> None:
    settings, db = _setup(tmp_path)
    parent = _run(settings, db, tmp_path)
    parent_problems = [p.id for p in ProblemRepo(db).list_for_run(parent)]

    settings.tickets.channel = "phone"
    plan = CasePlan(cases=[CasePlanEntry(problem=parent_problems[1])])
    child = _run(settings, db, tmp_path, case_plan=plan, problems_from=parent)

    run = RunRepo(db).get(child)
    assert (run.phase, run.parent_run_id) == ("phase2", parent)
    assert ProblemRepo(db).list_for_run(child) == []  # no new problems were generated
    with db.connect() as conn:
        lineage = conn.execute(
            "SELECT problem_id FROM lineage WHERE run_id = ?", (child,)
        ).fetchall()
        phase1_calls = conn.execute(
            "SELECT COUNT(*) AS n FROM agent_traces WHERE run_id = ? AND artifact_type = 'problem'",
            (child,),
        ).fetchone()["n"]
        channels = {
            r["channel"]
            for r in conn.execute("SELECT channel FROM resolutions WHERE run_id = ?", (child,))
        }
    assert [r["problem_id"] for r in lineage] == [parent_problems[1]]
    assert phase1_calls == 0
    assert channels == {"phone"}
    # The reused problem is the ground truth of the child's case: it is counted and exported.
    assert json.loads(run.stats_json or "{}")["problem_count"] == 1
    export_run_to_jsonl(db, child, out_dir=tmp_path / "exports")
    exported = (tmp_path / "exports" / child / "problems.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in exported] == [parent_problems[1]]


def test_cli_plan_loading_checks_problem_pins(tmp_path: Path) -> None:
    settings, _db = _setup(tmp_path)
    plan = tmp_path / "cases.yaml"
    plan.write_text("cases:\n  - problem: 7\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="problem index 7"):
        _load_plan(plan, None, settings)
    plan.write_text("cases:\n  - problem: some-id\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="pin an index instead"):
        _load_plan(plan, None, settings)
    plan.write_text("cases:\n  - contacts: 3\n    end_modes: [dropped]\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="needs 2 entries"):
        _load_plan(plan, None, settings)
    plan.write_text("cases:\n  - tone: polite\n  - problem: 1\n", encoding="utf-8")
    assert len(_load_plan(plan, None, settings).cases) == 2
    assert settings.tickets.total == 2


def test_cli_problems_from_checks_the_parent_run(tmp_path: Path) -> None:
    settings, db = _setup(tmp_path)
    with pytest.raises(typer.BadParameter, match="no run 'nope'"):
        _reused_problem_ids(db, "nope", settings)
    parent = _run(settings, db, tmp_path)
    assert len(_reused_problem_ids(db, parent, settings)) == 3
    settings.seeds.company = "norrholt"  # the parent was generated for kalvora
    with pytest.raises(typer.BadParameter, match="--company kalvora"):
        _reused_problem_ids(db, parent, settings)
