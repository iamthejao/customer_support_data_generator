from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from csfd.phases.phase1_kb.nodes import persist_article_node, persist_problem_node
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBArticleDraft,
    KBState,
    ProblemDraft,
)
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import KBArticleRepo, ProblemRepo, RunRecord, RunRepo


def _bootstrap(tmp_db_path: Path) -> tuple[Database, str]:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(
        RunRecord(
            id=run_id,
            phase="phase1",
            parent_run_id=None,
            status="running",
            started_at=datetime.now(UTC),
            completed_at=None,
            run_seed=1,
            pipeline_version="0.1.0",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    return db, run_id


def _state(run_id: str, problem: CommittedProblem) -> KBState:
    return KBState(
        run_id=run_id,
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        scenarios=ScenarioCatalogue(),
        problems_committed=[problem],
    )


def test_persist_problem_node_writes_row(tmp_db_path: Path) -> None:
    db, run_id = _bootstrap(tmp_db_path)
    pc = CommittedProblem(
        id=str(uuid4()),
        draft=ProblemDraft(title="t", description="d", category="c", severity="low"),
        has_kb=False,
        coverage_reasoning="r",
        coverage_confidence="medium",
    )
    state = _state(run_id, pc)
    persist_problem_node(state, db=db, problem=pc)
    rows = ProblemRepo(db).list_for_run(run_id)
    assert len(rows) == 1
    assert rows[0].id == pc.id
    assert rows[0].has_kb is False


def test_persist_article_node_writes_row(tmp_db_path: Path) -> None:
    db, run_id = _bootstrap(tmp_db_path)
    pc = CommittedProblem(
        id=str(uuid4()),
        draft=ProblemDraft(title="t", description="d", category="c", severity="low"),
        has_kb=True,
        coverage_reasoning="r",
        coverage_confidence="high",
    )
    state = _state(run_id, pc)
    persist_problem_node(state, db=db, problem=pc)
    article = CommittedArticle(
        id=str(uuid4()),
        problem_id=pc.id,
        draft=KBArticleDraft(
            title="x",
            content_markdown="y",
            troubleshooting_steps=[{"step": "a", "expected_result": "b"}],
        ),
    )
    persist_article_node(state, db=db, article=article)
    got = KBArticleRepo(db).get_by_problem(pc.id)
    assert got is not None
    assert got.title == "x"
