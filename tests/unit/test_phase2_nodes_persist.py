from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from csfd.phases.phase2_cases.nodes import (
    persist_ticket_node,
    persist_turn_node,
)
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CommittedTurn,
    CustomerPersona,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.seeds.company import CompanyProfile
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
    TurnRepo,
)


def _bootstrap(tmp_db_path: Path) -> tuple[Database, str, str]:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    parent_run_id = str(uuid4())
    run_id = str(uuid4())
    for rid, phase in [(parent_run_id, "phase1"), (run_id, "phase2")]:
        RunRepo(db).create(
            RunRecord(
                id=rid,
                phase=phase,
                parent_run_id=parent_run_id if phase == "phase2" else None,
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
            category="c",
            severity="low",
            has_kb=False,
            coverage_reasoning="r",
            coverage_confidence="low",
            metadata_json=None,
            quality_flag=None,
            unresolved_issues_json=None,
            created_at=datetime.now(UTC),
        )
    )
    return db, run_id, pid


def _state(run_id: str, ticket: CommittedTicket) -> TicketState:
    return TicketState(
        run_id=run_id,
        parent_run_id=str(uuid4()),
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# A"),
        current_ticket=ticket,
    )


def test_persist_ticket_node_writes_row(tmp_db_path: Path) -> None:
    db, run_id, pid = _bootstrap(tmp_db_path)
    ticket = CommittedTicket(
        id=str(uuid4()),
        draft=TicketDraft(
            problem_id=pid,
            kb_article_id=None,
            ticket_type="l3",
            priority="high",
            subject="x",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="n"),
            agent_persona=AgentPersona(name="A", tier="l3", expertise=""),
        ),
    )
    state = _state(run_id, ticket)
    persist_ticket_node(state, db=db, ticket=ticket)
    persist_ticket_node(state, db=db, ticket=ticket)  # idempotent
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, ticket_type FROM tickets WHERE id = ?",
            (ticket.id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["ticket_type"] == "l3"


def test_persist_turn_node_writes_row(tmp_db_path: Path) -> None:
    db, run_id, pid = _bootstrap(tmp_db_path)
    ticket = CommittedTicket(
        id=str(uuid4()),
        draft=TicketDraft(
            problem_id=pid,
            kb_article_id=None,
            ticket_type="l3",
            priority="high",
            subject="x",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="n"),
            agent_persona=AgentPersona(name="A", tier="l3", expertise=""),
        ),
    )
    state = _state(run_id, ticket)
    persist_ticket_node(state, db=db, ticket=ticket)

    turn = CommittedTurn(
        id=str(uuid4()),
        ticket_id=ticket.id,
        turn_index=0,
        draft=TurnDraft(
            speaker="customer",
            content="help",
            intent="question",
            noise_applied=True,
            noise_type="typos_informal_phrasing",
        ),
    )
    persist_turn_node(state, db=db, turn=turn)
    rows = TurnRepo(db).list_for_ticket(ticket.id)
    assert len(rows) == 1
    assert rows[0].turn_index == 0
    assert rows[0].noise_applied is True
    assert rows[0].noise_type == "typos_informal_phrasing"
