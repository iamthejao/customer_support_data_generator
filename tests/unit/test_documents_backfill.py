"""The back-fill's view of a run: registry, problem models, coverage and answer sources."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from csfd.diagnosis import CandidateCause, DiagnosisPlan
from csfd.documents.backfill import build_run_documents, load_context
from csfd.documents.build import BUILDERS
from csfd.documents.registry import build_registry
from csfd.documents.sidecar import render_markdown
from csfd.documents.troubleshooting import TroubleshootingBuilder
from csfd.seeds.company import parse_company_seed
from csfd.storage.db import Database
from csfd.storage.db_async import AsyncDatabase
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.product_facts import ProductFactsRepo
from csfd.storage.repository import (
    LineageRecord,
    LineageRepo,
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
)

RUN = "run-bf"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "registry_company_seed.md"
START = datetime(2026, 3, 1, tzinfo=UTC)


def _plan(cause: str) -> dict[str, object]:
    return DiagnosisPlan(
        candidate_causes=[CandidateCause(cause=cause, is_root_cause=True)]
    ).model_dump()


def _db(
    tmp_path: Path, *, calendar: bool = True, plan0: dict[str, object] | None = None
) -> Database:
    seeds = tmp_path / "seeds" / "acme"
    seeds.mkdir(parents=True)
    seeds.joinpath("company_seed.md").write_text(FIXTURE.read_text(), encoding="utf-8")
    db = Database(path=tmp_path / "runs.sqlite")
    apply_migrations(db)
    snapshot: dict[str, object] = {"company": {"name": "Acme Cloud", "seed": "acme"}}
    if calendar:
        snapshot["tickets"] = {"calendar": {"start": START.isoformat()}}
    RunRepo(db).create(
        RunRecord(
            id=RUN,
            phase="full",
            parent_run_id=None,
            status="completed",
            started_at=datetime(2026, 5, 5, tzinfo=UTC),
            completed_at=None,
            run_seed=9,
            pipeline_version="test",
            git_sha=None,
            config_snapshot_json=json.dumps(snapshot),
            stats_json=None,
            error_summary=None,
        )
    )
    # p0 cites a registry part; p1 names nothing, its cases ran on one model;
    # p2 names nothing, its cases ran on two models.
    texts = ["The AC-PSU-001 smells burnt", "It stops", "It stops"]
    case_models = [["AX-100"], ["AX-200", "AX-200"], ["AX-100", "AX-200"]]
    slot = 0
    for i, (text, models) in enumerate(zip(texts, case_models, strict=True)):
        pid = f"{RUN}:p:{i:04d}"
        ProblemRepo(db).create(
            ProblemRecord(
                id=pid,
                run_id=RUN,
                title="t",
                summary=text,
                background="b",
                category="c",
                complexity="simple",
                resolution_hints={},
                quality_flag=None,
                created_at=START,
                diagnosis_plan=plan0 if i == 0 and plan0 else _plan(f"cause {i}"),
            )
        )
        for model in models:
            LineageRepo(db).create(
                LineageRecord(
                    ticket_uid=f"{RUN}:{slot:06d}",
                    run_id=RUN,
                    slot_index=slot,
                    problem_id=pid,
                    ticket_type="l1",
                    customer_tier="standard",
                    customer_tone="neutral",
                    incoming_request_id=None,
                    resolution_id=None,
                    created_at=START,
                    case_facts={
                        "asset_model": model,
                        "customer_company": "Lab",
                        "site_city": "Bern",
                        "site_country": "Switzerland",
                    },
                )
            )
            slot += 1
    return db


def test_the_troubleshooting_builder_is_registered() -> None:
    assert BUILDERS["troubleshooting"] is TroubleshootingBuilder


def test_a_run_without_a_stored_registry_gets_the_code_drawn_one(tmp_path: Path) -> None:
    db = _db(tmp_path)
    ctx = load_context(db, RUN, seeds_dir=tmp_path / "seeds")
    company = parse_company_seed(FIXTURE)
    assert ctx.registry == build_registry(company, seed=9, calendar_start=START, names={})
    assert ctx.problem_models == {
        f"{RUN}:p:0000": "AX-100",  # holds the registry part it cites
        f"{RUN}:p:0001": "AX-200",  # the one model its cases ran on
        f"{RUN}:p:0002": None,
    }


def test_a_run_without_a_stored_registry_leaves_out_numbers_it_lacks(tmp_path: Path) -> None:
    """Problems of a run generated with documents disabled were written without the registry."""
    plan = DiagnosisPlan(
        candidate_causes=[
            CandidateCause(
                cause="Failed rear fan",
                is_root_cause=True,
                resolution_steps=["Clear code F-45, then replace the AC-PSU-999 fan"],
                parts=["AC-PSU-999 Fan, rear", "AC-PSU-001 Power supply unit, 240 W"],
            )
        ]
    ).model_dump()
    db = _db(tmp_path, plan0=plan)
    result = build_run_documents(
        db,
        RUN,
        [TroubleshootingBuilder()],
        seeds_dir=tmp_path / "seeds",
        coverage_proportions={"documented": 1.0},
    )
    kb = next(b.document for b in result.documents if b.document.problem_id == f"{RUN}:p:0000")
    text = render_markdown(kb, {})
    assert "F-45" not in text and "AC-PSU-999" not in text
    assert "Clear code, then replace the fan" in text
    assert "Fan, rear" in text and "AC-PSU-001" in text


async def test_a_stored_registry_wins(tmp_path: Path) -> None:
    db = _db(tmp_path)
    company = parse_company_seed(FIXTURE)
    stored = build_registry(company, seed=1234, calendar_start=START, names={})
    await ProductFactsRepo(db).areplace(AsyncDatabase(db.path), RUN, stored)
    ctx = load_context(db, RUN, seeds_dir=tmp_path / "seeds")
    assert ctx.registry == stored


@pytest.mark.parametrize(
    ("proportions", "source"),
    [
        ({"documented": 1.0}, "documents"),
        ({"partial": 1.0}, "partial"),
        ({"agent_only": 1.0}, "agent_knowledge"),
    ],
)
def test_coverage_and_answer_sources(
    tmp_path: Path, proportions: dict[str, float], source: str
) -> None:
    db = _db(tmp_path)
    ctx = load_context(
        db,
        RUN,
        seeds_dir=tmp_path / "seeds",
        coverage_proportions=proportions,  # type: ignore[arg-type]
    )
    assert ctx.coverage[f"{RUN}:p:0001"] == {"cause 1": next(iter(proportions))}
    assert set(ctx.answer_sources.values()) == {source}
    assert len(ctx.answer_sources) == 5


def test_a_run_without_a_calendar_dates_documents_before_its_start(tmp_path: Path) -> None:
    ctx = load_context(_db(tmp_path, calendar=False), RUN, seeds_dir=tmp_path / "seeds")
    assert ctx.calendar_start == datetime(2026, 5, 5, tzinfo=UTC)
