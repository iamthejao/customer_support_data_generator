"""Integration test for the runtime-wired Phase 1 single-problem graph."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.graph.runtime_phase1 import build_runtime_phase1_graph
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.state import (
    CoverageDecision,
    KBArticleDraft,
    KBState,
    ProblemDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import parse_company_seed
from csfd.seeds.scenarios import parse_scenarios_seed
from csfd.settings import AgentLLMConfig
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRepo,
    ProblemRepo,
    RunRecord,
    RunRepo,
)


def _factory(canned: dict[type, Any]) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.5)
    agent_names = (
        "problem_brainstorm",
        "problem_consistency",
        "problem_background",
        "problem_scenario",
        "coverage_judge",
        "article_writer",
        "article_consistency",
        "article_background",
        "article_scenario",
    )
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda _c: FakeChatModel(structured=canned),
        agent_configs={n: cfg for n in agent_names},
    )


@pytest.mark.asyncio
async def test_runtime_phase1_full_loop_persists_problem_and_article(
    tmp_db_path: Path,
) -> None:
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

    factory = _factory(
        {
            ProblemDraft: ProblemDraft(
                title="Cannot reset password",
                description="Customer locked out",
                category="auth",
                severity="medium",
            ),
            CoverageDecision: CoverageDecision(
                has_kb=True,
                reasoning="ok",
                confidence="high",
            ),
            KBArticleDraft: KBArticleDraft(
                title="Recover login",
                content_markdown="## Step 1",
                troubleshooting_steps=[{"step": "x", "expected_result": "y"}],
            ),
            Verdict: Verdict(checker="x", passed=True),
        }
    )

    graph = build_runtime_phase1_graph(
        factory=factory,
        db=db,
        max_retries=2,
        kb_target_rate=1.0,
    )
    initial = KBState(
        run_id=run_id,
        run_seed=1,
        company=parse_company_seed(Path("tests/fixtures/tiny_company_seed.md")),
        scenarios=parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md")),
    )
    await graph.ainvoke(initial)

    problems = ProblemRepo(db).list_for_run(run_id)
    assert len(problems) == 1
    assert problems[0].has_kb is True
    article = KBArticleRepo(db).get_by_problem(problems[0].id)
    assert article is not None
    assert article.title == "Recover login"
