import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
    TicketRecord,
    TicketRepo,
    TurnRecord,
    TurnRepo,
)


def _bootstrap(tmp_db_path: Path) -> tuple[Database, str, str]:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase2", parent_run_id=None, status="running",
        started_at=datetime.now(UTC), completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None, config_snapshot_json="{}",
        stats_json=None, error_summary=None,
    ))
    pid = str(uuid4())
    ProblemRepo(db).create(ProblemRecord(
        id=pid, run_id=run_id, title="t", description="d",
        category="c", severity="low", has_kb=False,
        coverage_reasoning=None, coverage_confidence=None,
        metadata_json=None, quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(UTC),
    ))
    return db, run_id, pid


def test_create_ticket_and_turns(tmp_db_path: Path) -> None:
    db, run_id, pid = _bootstrap(tmp_db_path)
    t_repo = TicketRepo(db)
    tu_repo = TurnRepo(db)

    tid = str(uuid4())
    t_repo.create(TicketRecord(
        id=tid, run_id=run_id, problem_id=pid, kb_article_id=None,
        ticket_type="l3", priority="high", status="resolved",
        subject="My printer is on fire",
        customer_persona_json=json.dumps({"name": "Alex"}),
        agent_persona_json=json.dumps({"name": "Tier3-Specialist"}),
        ground_truth_json=None, metadata_json=None,
        quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(UTC), resolved_at=None,
    ))
    for i, (speaker, content) in enumerate(
        [("customer", "help!"), ("agent", "describe?"), ("customer", "smoke")]
    ):
        tu_repo.create(TurnRecord(
            id=str(uuid4()), ticket_id=tid, turn_index=i,
            speaker=speaker, speaker_persona=None, content=content,
            intent=None, kb_references_json=None,
            noise_applied=False, noise_type=None,
            quality_flag=None,
            created_at=datetime.now(UTC),
        ))
    turns = tu_repo.list_for_ticket(tid)
    assert len(turns) == 3
    assert [t.turn_index for t in turns] == [0, 1, 2]


def test_turn_unique_index_per_ticket(tmp_db_path: Path) -> None:
    db, run_id, pid = _bootstrap(tmp_db_path)
    t_repo = TicketRepo(db)
    tu_repo = TurnRepo(db)
    tid = str(uuid4())
    t_repo.create(TicketRecord(
        id=tid, run_id=run_id, problem_id=pid, kb_article_id=None,
        ticket_type="l1", priority="low", status="resolved", subject="s",
        customer_persona_json="{}", agent_persona_json="{}",
        ground_truth_json=None, metadata_json=None,
        quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(UTC), resolved_at=None,
    ))
    base_turn = TurnRecord(
        id=str(uuid4()), ticket_id=tid, turn_index=0,
        speaker="customer", speaker_persona=None, content="x",
        intent=None, kb_references_json=None,
        noise_applied=False, noise_type=None, quality_flag=None,
        created_at=datetime.now(UTC),
    )
    tu_repo.create(base_turn)
    duplicate = TurnRecord(
        id=str(uuid4()), ticket_id=tid, turn_index=0,  # same turn_index
        speaker="customer", speaker_persona=None, content="x",
        intent=None, kb_references_json=None,
        noise_applied=False, noise_type=None, quality_flag=None,
        created_at=datetime.now(UTC),
    )
    tu_repo.create(duplicate)
    assert len(tu_repo.list_for_ticket(tid)) == 1
