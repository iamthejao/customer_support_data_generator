import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import (
    article_writer_node,
    coverage_decider_node,
    persist_article_node,
    persist_problem_node,
    problem_background_check_node,
    problem_brainstorm_node,
    problem_consistency_check_node,
    problem_scenario_check_node,
)
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
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


def _factory(canned: dict[type, object]) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    cfg = AgentLLMConfig(
        provider="anthropic",
        model="claude-haiku-4-5",
        temperature=0.5,
    )
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
        llm_builder=lambda c: FakeChatModel(structured=canned),
        agent_configs={n: cfg for n in agent_names},
    )


@pytest.mark.asyncio
async def test_phase1_full_pipeline_on_tiny_seeds(tmp_db_path: Path) -> None:
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

    canned_problem = ProblemDraft(
        title="Cannot reset password",
        description="Customer locked out after failed attempts",
        category="Authentication",
        severity="medium",
    )
    canned_decision = CoverageDecision(
        has_kb=True,
        reasoning="Common",
        confidence="high",
    )
    canned_article = KBArticleDraft(
        title="How to recover login",
        content_markdown="## Step 1\nReset via email link.",
        troubleshooting_steps=[
            {"step": "Click 'forgot password'", "expected_result": "email sent"},
        ],
    )
    canned_pass = Verdict(checker="x", passed=True)

    factory = _factory(
        {
            ProblemDraft: canned_problem,
            CoverageDecision: canned_decision,
            KBArticleDraft: canned_article,
            Verdict: canned_pass,
        }
    )

    state = KBState(
        run_id=run_id,
        run_seed=1,
        company=parse_company_seed(Path("tests/fixtures/tiny_company_seed.md")),
        scenarios=parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md")),
    )

    # 1. Brainstorm
    update = await problem_brainstorm_node(state, factory=factory)
    state = state.model_copy(update=update)
    assert state.current_problem_draft is not None

    # 2. Three checkers in parallel
    payload = {"current_problem_draft": state.current_problem_draft}
    verdict_updates = await asyncio.gather(
        problem_consistency_check_node(payload, factory=factory),
        problem_background_check_node(payload, factory=factory),
        problem_scenario_check_node(payload, factory=factory),
    )
    state = state.model_copy(
        update={
            "verdicts": [v["verdicts"][0] for v in verdict_updates],
        }
    )
    assert all(v.passed for v in state.verdicts)

    # 3. Commit problem (coverage to be decided)
    pid = str(uuid4())
    pc = CommittedProblem(
        id=pid,
        draft=state.current_problem_draft,
        has_kb=False,
        coverage_reasoning="",
        coverage_confidence="low",
    )
    state = state.model_copy(
        update={
            "problems_committed": [pc],
            "verdicts": [],
        }
    )

    # 4. Coverage decision (force-target 1.0 = always covered)
    update = await coverage_decider_node(state, factory=factory, target_rate=1.0)
    state = state.model_copy(update=update)
    assert state.problems_committed[0].has_kb is True

    # 5. Persist problem
    persist_problem_node(state, db=db, problem=state.problems_committed[0])

    # 6. Write article
    update = await article_writer_node(
        state,
        factory=factory,
        problem=state.problems_committed[0],
    )
    state = state.model_copy(update=update)
    assert state.current_article_draft is not None

    # 7. Persist article
    article = CommittedArticle(
        id=str(uuid4()),
        problem_id=pid,
        draft=state.current_article_draft,
    )
    persist_article_node(state, db=db, article=article)

    # 8. Verify SQLite
    problems = ProblemRepo(db).list_for_run(run_id)
    assert len(problems) == 1
    assert problems[0].has_kb is True
    article_row = KBArticleRepo(db).get_by_problem(pid)
    assert article_row is not None
    assert article_row.title == "How to recover login"
