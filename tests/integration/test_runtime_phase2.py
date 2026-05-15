"""Integration test for the runtime-wired Phase 2 single-turn graph."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from csfd.agents.base import Verdict
from csfd.agents.creative_noise import TurnModification
from csfd.agents.factory import AgentFactory
from csfd.graph.runtime_phase2 import build_runtime_phase2_graph
from csfd.models.fake import FakeChatModel
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CustomerPersona,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.settings import AgentLLMConfig
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
    TurnRepo,
)


def _factory(canned: dict[type, Any]) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.7)
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda _c: FakeChatModel(structured=canned),
        agent_configs={
            n: cfg
            for n in (
                "turn_writer",
                "turn_consistency",
                "turn_background",
                "turn_scenario",
                "creative_noise",
            )
        },
    )


@pytest.mark.asyncio
async def test_runtime_phase2_single_turn_persists_with_noise(
    tmp_db_path: Path,
) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    parent_run_id = str(uuid4())
    run_id = str(uuid4())
    for rid, phase, parent in [
        (parent_run_id, "phase1", None),
        (run_id, "phase2", parent_run_id),
    ]:
        RunRepo(db).create(
            RunRecord(
                id=rid,
                phase=phase,
                parent_run_id=parent,
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
    pid = str(uuid4())
    ProblemRepo(db).create(
        ProblemRecord(
            id=pid,
            run_id=parent_run_id,
            title="t",
            description="d",
            category="auth",
            severity="medium",
            has_kb=False,
            coverage_reasoning="r",
            coverage_confidence="low",
            metadata_json=None,
            quality_flag=None,
            unresolved_issues_json=None,
            created_at=datetime.now(UTC),
        )
    )

    factory = _factory(
        {
            TurnDraft: TurnDraft(
                speaker="agent",
                content="Diagnosing now.",
                intent="clarification",
            ),
            Verdict: Verdict(checker="x", passed=True),
            TurnModification: TurnModification(
                modified_content="diagnosing now (informal)",
                noise_type="typos_informal_phrasing",
                rationale="casual",
            ),
        }
    )

    ticket = CommittedTicket(
        id=str(uuid4()),
        draft=TicketDraft(
            problem_id=pid,
            kb_article_id=None,
            ticket_type="l3",
            priority="high",
            subject="t",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="neutral"),
            agent_persona=AgentPersona(name="A", tier="l3", expertise=""),
        ),
    )

    graph = build_runtime_phase2_graph(
        factory=factory,
        db=db,
        max_retries=2,
        noise_probability=1.0,
        noise_type_weights={"typos_informal_phrasing": 1.0},
    )
    initial = TicketState(
        run_id=run_id,
        parent_run_id=parent_run_id,
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        current_ticket=ticket,
    )
    await graph.ainvoke(initial)

    rows = TurnRepo(db).list_for_ticket(ticket.id)
    assert len(rows) == 1
    assert rows[0].noise_applied is True
    assert rows[0].noise_type == "typos_informal_phrasing"
