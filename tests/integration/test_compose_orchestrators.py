"""Integration tests for the Phase 1 / Phase 2 compose orchestrators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from csfd.agents.base import Verdict
from csfd.agents.creative_noise import TurnModification
from csfd.agents.factory import AgentFactory
from csfd.graph.compose import run_phase1, run_phase2
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.state import (
    CoverageDecision,
    KBArticleDraft,
    ProblemDraft,
)
from csfd.phases.phase2_cases.state import TurnDraft
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import parse_company_seed
from csfd.seeds.scenarios import parse_scenarios_seed
from csfd.settings import (
    AgentLLMConfig,
    Phase1Config,
    Phase2Config,
    TicketsPerProblem,
    TicketTypeWeights,
)
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import ProblemRepo


def _all_agents_factory(canned: dict[type, Any]) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.5)
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda _c: FakeChatModel(structured=canned),
        agent_configs={
            n: cfg
            for n in (
                "problem_brainstorm",
                "problem_consistency",
                "problem_background",
                "problem_scenario",
                "coverage_judge",
                "article_writer",
                "article_consistency",
                "article_background",
                "article_scenario",
                "turn_writer",
                "turn_consistency",
                "turn_background",
                "turn_scenario",
                "creative_noise",
            )
        },
    )


@pytest.mark.asyncio
async def test_run_phase1_generates_problems(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    factory = _all_agents_factory(
        {
            ProblemDraft: ProblemDraft(
                title="P",
                description="d",
                category="c",
                severity="low",
            ),
            CoverageDecision: CoverageDecision(has_kb=True, reasoning="ok", confidence="high"),
            KBArticleDraft: KBArticleDraft(
                title="A",
                content_markdown="x",
                troubleshooting_steps=[{"step": "s", "expected_result": "r"}],
            ),
            Verdict: Verdict(checker="x", passed=True),
        }
    )
    cfg = Phase1Config(
        problem_count=3,
        kb_coverage_target_rate=1.0,
        dedup_similarity_threshold=0.999,
        dedup_method="lexical",
    )
    run_id = await run_phase1(
        factory=factory,
        db=db,
        phase1_cfg=cfg,
        run_seed=42,
        max_retries=2,
        company=parse_company_seed(Path("tests/fixtures/tiny_company_seed.md")),
        scenarios=parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md")),
        pipeline_version="0.1.0",
    )
    problems = ProblemRepo(db).list_for_run(run_id)
    assert len(problems) >= 1


@pytest.mark.asyncio
async def test_run_phase2_chains_off_phase1_all_no_kb_become_l3(
    tmp_db_path: Path,
) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    factory = _all_agents_factory(
        {
            ProblemDraft: ProblemDraft(
                title="P",
                description="d",
                category="c",
                severity="low",
            ),
            CoverageDecision: CoverageDecision(has_kb=False, reasoning="no", confidence="high"),
            KBArticleDraft: KBArticleDraft(
                title="A",
                content_markdown="x",
                troubleshooting_steps=[{"step": "s", "expected_result": "r"}],
            ),
            Verdict: Verdict(checker="x", passed=True),
            TurnDraft: TurnDraft(speaker="agent", content="hi", intent="resolution"),
            TurnModification: TurnModification(
                modified_content="hi (informal)",
                noise_type="typos_informal_phrasing",
                rationale="casual",
            ),
        }
    )
    p1_cfg = Phase1Config(
        problem_count=2,
        kb_coverage_target_rate=0.0,
        dedup_similarity_threshold=0.999,
        dedup_method="lexical",
    )
    p2_cfg = Phase2Config(
        tickets_per_problem=TicketsPerProblem(has_kb=(1, 1), no_kb=(1, 1)),
        ticket_type_weights=TicketTypeWeights(has_kb={"l1": 1.0}, no_kb={"l3": 1.0}),
        creative_noise_probability=0.0,
        noise_type_weights={"typos_informal_phrasing": 1.0},
        min_turns_per_ticket=1,
        max_turns_per_ticket=1,
    )
    p1_run = await run_phase1(
        factory=factory,
        db=db,
        phase1_cfg=p1_cfg,
        run_seed=1,
        max_retries=2,
        company=parse_company_seed(Path("tests/fixtures/tiny_company_seed.md")),
        scenarios=parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md")),
        pipeline_version="0.1.0",
    )
    p2_run = await run_phase2(
        factory=factory,
        db=db,
        phase2_cfg=p2_cfg,
        run_seed=1,
        max_retries=2,
        parent_run_id=p1_run,
        company=parse_company_seed(Path("tests/fixtures/tiny_company_seed.md")),
        pipeline_version="0.1.0",
    )
    with db.connect() as conn:
        tickets = conn.execute(
            "SELECT id, ticket_type FROM tickets WHERE run_id = ?", (p2_run,)
        ).fetchall()
    assert len(tickets) >= 1
    for row in tickets:
        assert row["ticket_type"] == "l3"
