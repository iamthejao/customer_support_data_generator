import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.base import Verdict
from csfd.agents.creative_noise import TurnModification
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase2_cases.nodes import (
    creative_noise_gate,
    creative_noise_node,
    load_kb_node,
    persist_ticket_node,
    persist_turn_node,
    ticket_init_node,
    ticket_sampler_node,
    turn_background_check_node,
    turn_consistency_check_node,
    turn_scenario_check_node,
    turn_writer_node,
)
from csfd.phases.phase2_cases.state import (
    CommittedTurn,
    TicketState,
    TurnDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.settings import (
    AgentLLMConfig,
    Phase2Config,
    TicketsPerProblem,
    TicketTypeWeights,
)
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRecord,
    KBArticleRepo,
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
    TurnRepo,
)
from csfd.utils.hashing import short_hash


def _seed_phase1_db(tmp_db_path: Path) -> tuple[Database, str, str, str]:
    """Create two problems (1 with KB, 1 without) in a parent phase1 run."""
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
    p_kb = str(uuid4())
    p_no = str(uuid4())
    for pid, has_kb in [(p_kb, True), (p_no, False)]:
        ProblemRepo(db).create(
            ProblemRecord(
                id=pid,
                run_id=parent_run_id,
                title=f"P-{pid[:4]}",
                description="d",
                category="auth",
                severity="medium",
                has_kb=has_kb,
                coverage_reasoning="r",
                coverage_confidence="high",
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
            problem_id=p_kb,
            title="Recover login",
            content_markdown="## Step 1",
            content_hash=short_hash("## Step 1"),
            troubleshooting_steps_json="[]",
            prerequisites_json=None,
            metadata_json=None,
            version=1,
            quality_flag=None,
            unresolved_issues_json=None,
            created_at=datetime.now(UTC),
        )
    )
    return db, run_id, parent_run_id, p_no


def _factory_for_e2e(
    canned_turn: TurnDraft,
    canned_verdict: Verdict,
    canned_noise: TurnModification,
) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    cfg = AgentLLMConfig(
        provider="anthropic",
        model="claude-haiku-4-5",
        temperature=0.7,
    )
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda _c: FakeChatModel(
            structured={
                TurnDraft: canned_turn,
                Verdict: canned_verdict,
                TurnModification: canned_noise,
            }
        ),
        agent_configs={
            "turn_writer": cfg,
            "turn_consistency": cfg,
            "turn_background": cfg,
            "turn_scenario": cfg,
            "creative_noise": cfg,
        },
    )


def _phase2_cfg(noise_prob: float) -> Phase2Config:
    return Phase2Config(
        tickets_per_problem=TicketsPerProblem(has_kb=(1, 1), no_kb=(1, 1)),
        ticket_type_weights=TicketTypeWeights(
            has_kb={"docs_request": 0.0, "l1": 1.0, "l2": 0.0, "l3": 0.0},
            no_kb={"l3": 1.0},
        ),
        creative_noise_probability=noise_prob,
        noise_type_weights={"typos_informal_phrasing": 1.0},
        min_turns_per_ticket=2,
        max_turns_per_ticket=2,
    )


async def _run_ticket(
    state: TicketState,
    *,
    factory: AgentFactory,
    db: Database,
    cfg: Phase2Config,
) -> TicketState:
    """Run the inner per-ticket loop until ticket_done."""
    while len(state.turns_committed) < cfg.max_turns_per_ticket:
        # 1. Write a turn
        update = await turn_writer_node(state, factory=factory)
        state = state.model_copy(update=update)
        # 2. Creative noise gate
        gate = creative_noise_gate(state, probability=cfg.creative_noise_probability)
        if gate == "apply_noise":
            update = await creative_noise_node(
                state,
                factory=factory,
                noise_type_weights=cfg.noise_type_weights,
            )
            state = state.model_copy(update=update)
        # 3. Run 3 checkers in parallel
        assert state.current_turn_draft is not None
        payload = {"current_turn_draft": state.current_turn_draft.model_dump()}
        verdicts = await asyncio.gather(
            turn_consistency_check_node(payload, factory=factory),
            turn_background_check_node(payload, factory=factory),
            turn_scenario_check_node(payload, factory=factory),
        )
        state = state.model_copy(
            update={
                "verdicts": [v["verdicts"][0] for v in verdicts],
            }
        )
        assert all(v.passed for v in state.verdicts)
        # 4. Persist turn
        assert state.current_turn_draft is not None
        assert state.current_ticket is not None
        committed = CommittedTurn(
            id=str(uuid4()),
            ticket_id=state.current_ticket.id,
            turn_index=state.turn_index,
            draft=state.current_turn_draft,
        )
        persist_turn_node(state, db=db, turn=committed)
        state = state.model_copy(
            update={
                "turns_committed": [*state.turns_committed, committed],
                "current_turn_draft": None,
                "noise_applied": False,
                "noise_type": None,
                "turn_index": state.turn_index + 1,
                "verdicts": [],
            }
        )
    return state


@pytest.mark.asyncio
async def test_phase2_no_kb_forces_l3_and_noise_always_applied(
    tmp_db_path: Path,
) -> None:
    db, run_id, parent_run_id, p_no_id = _seed_phase1_db(tmp_db_path)
    canned_turn = TurnDraft(
        speaker="agent",
        content="Diagnosing now.",
        intent="clarification",
    )
    canned_verdict = Verdict(checker="x", passed=True)
    canned_noise = TurnModification(
        modified_content="diagnosing... brb",
        noise_type="typos_informal_phrasing",
        rationale="casual",
    )
    factory = _factory_for_e2e(canned_turn, canned_verdict, canned_noise)
    cfg = _phase2_cfg(noise_prob=1.0)

    state = TicketState(
        run_id=run_id,
        parent_run_id=parent_run_id,
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
    )
    state = state.model_copy(update=await load_kb_node(state, db=db))
    state = state.model_copy(
        update={
            "problem_pool": [p for p in state.problem_pool if p.id == p_no_id],
        }
    )

    update = await ticket_sampler_node(state, cfg=cfg)
    assert update["ticket_type"] == "l3"

    problem = state.problem_pool[0]
    update = await ticket_init_node(
        state,
        problem=problem,
        ticket_type="l3",
        kb_article_id=None,
    )
    state = state.model_copy(update=update)
    assert state.current_ticket is not None
    assert state.current_ticket.draft.ticket_type == "l3"

    persist_ticket_node(state, db=db, ticket=state.current_ticket)
    state = await _run_ticket(state, factory=factory, db=db, cfg=cfg)

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT ticket_type FROM tickets WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["ticket_type"] == "l3"

    assert state.current_ticket is not None
    turn_rows = TurnRepo(db).list_for_ticket(state.current_ticket.id)
    assert len(turn_rows) == 2
    for t in turn_rows:
        assert t.noise_applied is True
        assert t.noise_type == "typos_informal_phrasing"


@pytest.mark.asyncio
async def test_phase2_with_kb_uses_l1_and_no_noise_when_probability_zero(
    tmp_db_path: Path,
) -> None:
    db, run_id, parent_run_id, _p_no_id = _seed_phase1_db(tmp_db_path)
    canned_turn = TurnDraft(
        speaker="agent",
        content="Hi, let me help via our reset link.",
        intent="resolution",
    )
    canned_verdict = Verdict(checker="x", passed=True)
    canned_noise = TurnModification(
        modified_content="unused",
        noise_type="x",
        rationale="",
    )
    factory = _factory_for_e2e(canned_turn, canned_verdict, canned_noise)
    cfg = _phase2_cfg(noise_prob=0.0)

    state = TicketState(
        run_id=run_id,
        parent_run_id=parent_run_id,
        run_seed=2,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
    )
    state = state.model_copy(update=await load_kb_node(state, db=db))
    state = state.model_copy(
        update={
            "problem_pool": [p for p in state.problem_pool if p.has_kb],
        }
    )

    update = await ticket_sampler_node(state, cfg=cfg)
    assert update["ticket_type"] == "l1"

    problem = state.problem_pool[0]
    kb_article = state.kb_by_problem[problem.id]
    update = await ticket_init_node(
        state,
        problem=problem,
        ticket_type="l1",
        kb_article_id=kb_article.id,
    )
    state = state.model_copy(update=update)
    assert state.current_ticket is not None
    persist_ticket_node(state, db=db, ticket=state.current_ticket)
    state = await _run_ticket(state, factory=factory, db=db, cfg=cfg)

    assert state.current_ticket is not None
    turn_rows = TurnRepo(db).list_for_ticket(state.current_ticket.id)
    assert len(turn_rows) == 2
    for t in turn_rows:
        assert t.noise_applied is False
        assert t.noise_type is None
