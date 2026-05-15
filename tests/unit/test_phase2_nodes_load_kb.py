from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.phases.phase2_cases.nodes import load_kb_node
from csfd.phases.phase2_cases.state import TicketState
from csfd.seeds.company import CompanyProfile
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRecord,
    KBArticleRepo,
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
)
from csfd.utils.hashing import short_hash


def _bootstrap_phase1(tmp_db_path: Path) -> tuple[Database, str]:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    parent_run_id = str(uuid4())
    RunRepo(db).create(
        RunRecord(
            id=parent_run_id,
            phase="phase1",
            parent_run_id=None,
            status="completed",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            run_seed=1,
            pipeline_version="0.1.0",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    p_with_kb = str(uuid4())
    p_no_kb = str(uuid4())
    for pid, has_kb in [(p_with_kb, True), (p_no_kb, False)]:
        ProblemRepo(db).create(
            ProblemRecord(
                id=pid,
                run_id=parent_run_id,
                title=f"P-{pid[:4]}",
                description="d",
                category="c",
                severity="low",
                has_kb=has_kb,
                coverage_reasoning="r",
                coverage_confidence="medium",
                metadata_json=None,
                quality_flag=None,
                unresolved_issues_json=None,
                created_at=datetime.now(UTC),
            )
        )
    KBArticleRepo(db).create(
        KBArticleRecord(
            id=str(uuid4()),
            run_id=parent_run_id,
            problem_id=p_with_kb,
            title="Article",
            content_markdown="## Steps",
            content_hash=short_hash("## Steps"),
            troubleshooting_steps_json="[]",
            prerequisites_json=None,
            metadata_json=None,
            version=1,
            quality_flag=None,
            unresolved_issues_json=None,
            created_at=datetime.now(UTC),
        )
    )
    return db, parent_run_id


@pytest.mark.asyncio
async def test_load_kb_node_populates_pool_and_articles(tmp_db_path: Path) -> None:
    db, parent_run_id = _bootstrap_phase1(tmp_db_path)
    state = TicketState(
        run_id=str(uuid4()),
        parent_run_id=parent_run_id,
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
    )
    update = await load_kb_node(state, db=db)
    assert "problem_pool" in update
    assert "kb_by_problem" in update
    pool = update["problem_pool"]
    kbs = update["kb_by_problem"]
    assert isinstance(pool, list)
    assert isinstance(kbs, dict)
    assert len(pool) == 2
    assert len(kbs) == 1
    covered = [p for p in pool if p.has_kb]
    assert len(covered) == 1
    assert covered[0].id in kbs
