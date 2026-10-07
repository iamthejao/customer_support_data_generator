"""CLI: `csfd documents` builds troubleshooting documents for a finished run, end to end.

A run with two problems on the AX-100 (registry seed fixture), one case each,
back-filled with the shipped `troubleshooting` builder and exported. No model
is called.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from csfd.cli import app
from csfd.diagnosis import CandidateCause, DiagnosisPlan, DiagnosticCheck
from csfd.storage.db import Database
from csfd.storage.documents import DocumentRepo
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    IncomingRequestRecord,
    IncomingRequestRepo,
    LineageRecord,
    LineageRepo,
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
)

runner = CliRunner()
RUN_ID = "run-ts"
# With these causes and proportions, this seed documents one root cause and not the other.
RUN_SEED = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SEED = REPO_ROOT / "tests" / "fixtures" / "registry_company_seed.md"

PLANS = [
    DiagnosisPlan(
        candidate_causes=[
            CandidateCause(
                cause="Failed fan tray",
                is_root_cause=True,
                resolution_steps=["Replace the fan tray"],
                parts=["Fan tray, front"],
            ),
            CandidateCause(cause="Blocked air inlet", resolution_steps=["Clean the inlet"]),
        ],
        checks=[
            DiagnosticCheck(
                check="Fan LED", how_to_check="Look at the fan LED", finding="it blinks red"
            )
        ],
    ),
    DiagnosisPlan(
        candidate_causes=[
            CandidateCause(
                cause="Corrupt configuration",
                is_root_cause=True,
                resolution_steps=["Restore the last good configuration"],
            ),
            CandidateCause(cause="Full disk", resolution_steps=["Free disk space"]),
        ],
        checks=[
            DiagnosticCheck(
                check="Boot log", how_to_check="Read the last boot log", finding="parse error"
            )
        ],
    ),
]


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "config"
    (config / "profiles").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "config" / "default.yaml", config / "default.yaml")
    (config / "profiles" / "docs.yaml").write_text(
        "documents:\n  enabled: true\n  coverage_proportions:\n"
        "    documented: 0.5\n    partial: 0\n    agent_only: 0.5\n",
        encoding="utf-8",
    )
    seeds = tmp_path / "seeds" / "acme"
    seeds.mkdir(parents=True)
    shutil.copy(FIXTURE_SEED, seeds / "company_seed.md")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _seed(db_path: Path) -> Database:
    db = Database(path=db_path)
    apply_migrations(db)
    now = datetime.now(UTC)
    RunRepo(db).create(
        RunRecord(
            id=RUN_ID,
            phase="full",
            parent_run_id=None,
            status="completed",
            started_at=now,
            completed_at=now,
            run_seed=RUN_SEED,
            pipeline_version="test",
            git_sha=None,
            config_snapshot_json=json.dumps(
                {
                    "company": {"name": "Acme Cloud", "seed": "acme"},
                    "tickets": {"calendar": {"start": "2026-03-01T00:00:00+00:00"}},
                }
            ),
            stats_json=None,
            error_summary=None,
        )
    )
    for i, plan in enumerate(PLANS):
        problem_id = f"{RUN_ID}:p:{i:04d}"
        ProblemRepo(db).create(
            ProblemRecord(
                id=problem_id,
                run_id=RUN_ID,
                title=f"AX-100 problem {i}",
                summary="The AX-100 stops.",
                background="b",
                category="c",
                complexity="simple",
                resolution_hints={},
                quality_flag=None,
                created_at=now,
                symptoms=["The display shows F-01"] if i == 0 else ["It reboots"],
                diagnosis_plan=plan.model_dump(),
            )
        )
        ticket_uid = f"{RUN_ID}:{i:06d}"
        LineageRepo(db).create(
            LineageRecord(
                ticket_uid=ticket_uid,
                run_id=RUN_ID,
                slot_index=i,
                problem_id=problem_id,
                ticket_type="l1",
                customer_tier="standard",
                customer_tone="neutral",
                incoming_request_id=None,
                resolution_id=None,
                created_at=now,
                case_facts={
                    "asset_model": "AX-100",
                    "asset_serial": "AX100-00042-B",
                    "customer_company": "Lab",
                    "site_city": "Bern",
                    "site_country": "Switzerland",
                },
            )
        )
        ir_id = IncomingRequestRepo(db).create(
            IncomingRequestRecord(
                request_uid=f"{ticket_uid}:req",
                run_id=RUN_ID,
                problem_id=problem_id,
                ticket_type="l1",
                customer_name="Customer",
                customer_tier="standard",
                customer_tone="neutral",
                channel="email",
                subject=f"Problem {i}",
                body=f"Our AX-100 has problem {i}.",
                quality_flag=None,
                created_at=now,
                case_uid=ticket_uid,
            )
        )
        LineageRepo(db).update_links(ticket_uid, incoming_request_id=ir_id)
    return db


def _invoke(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_back_fill_and_export_troubleshooting_documents(workdir: Path) -> None:
    db_path = workdir / "runs.sqlite"
    db = _seed(db_path)
    code, out = _invoke("documents", RUN_ID, "--sqlite-path", str(db_path), "--profile", "docs")
    assert code == 0, out
    repo = DocumentRepo(db)
    sources = repo.answer_sources(RUN_ID)
    documented = next(uid for uid, s in sources.items() if s == "documents")
    agent_only = next(uid for uid, s in sources.items() if s == "agent_knowledge")
    docs = repo.list_for_run(RUN_ID)
    assert [(d.tier, d.doc_type) for d in docs] == [
        ("library", "service_manual"),
        ("problem", "kb_article"),
        ("problem", "kb_article"),  # the agent-only root cause's problem: its other cause
    ]
    assert docs[0].doc_number == "AC-SVC-100"  # the seed's service-manual number

    exports = workdir / "exports"
    code, out = _invoke(
        "export",
        RUN_ID,
        "--format",
        "documents",
        "--sqlite-path",
        str(db_path),
        "--out",
        str(exports),
    )
    assert code == 0, out
    root = exports / RUN_ID / "documents"
    for doc in docs:
        folder = root / "docs" / doc.folder_name
        for ext in ("docx", "pdf", "md", "json"):
            assert (folder / f"{doc.folder_name}.{ext}").is_file()

    queries = {q["_id"]: q["metadata"] for q in _rows(root / "rag" / "queries.jsonl")}
    assert queries[f"q:{documented}"]["answer_source"] == "documents"  # type: ignore[index]
    assert queries[f"q:{documented}"]["answerable_from_documents"] is True  # type: ignore[index]
    assert queries[f"q:{agent_only}"]["answer_source"] == "agent_knowledge"  # type: ignore[index]
    assert queries[f"q:{agent_only}"]["answerable_from_documents"] is False  # type: ignore[index]

    resolves = (root / "rag" / "qrels" / "test.resolves.tsv").read_text().splitlines()[1:]
    assert resolves and all(line.startswith(f"q:{documented}\t") for line in resolves)
    graded = (root / "rag" / "qrels" / "test.tsv").read_text().splitlines()[1:]
    assert any(line.startswith(f"q:{agent_only}\t") for line in graded)  # its checks still help
    detailed = _rows(root / "rag" / "qrels_detailed.jsonl")
    undocumented = [r for r in detailed if r["relation"] == "undocumented"]
    assert undocumented == [
        {
            "query-id": f"q:{agent_only}",
            "corpus-id": None,
            "score": 0,
            "relation": "undocumented",
            "basis": "root_cause_fix",
            "cause": "Corrupt configuration",
            "round_index": None,
        }
    ]
    corpus = (root / "rag" / "corpus.jsonl").read_text()
    assert "Corrupt configuration" not in corpus and "Restore the last good" not in corpus

    # Building and exporting again gives the same documents, links and files.
    tree = _tree(root)
    links = repo.links_for_run(RUN_ID)
    code, out = _invoke("documents", RUN_ID, "--sqlite-path", str(db_path), "--profile", "docs")
    assert code == 0, out
    assert repo.links_for_run(RUN_ID) == links
    code, out = _invoke(
        "export",
        RUN_ID,
        "--format",
        "documents",
        "--sqlite-path",
        str(db_path),
        "--out",
        str(exports),
    )
    assert code == 0, out
    again = _tree(root)
    assert {k: v for k, v in again.items() if k != "manifest.json"} == {
        k: v for k, v in tree.items() if k != "manifest.json"
    }


def test_a_database_from_the_previous_documents_schema_is_reported_stale(workdir: Path) -> None:
    db_path = workdir / "runs.sqlite"
    _seed(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        DROP TABLE document_links;
        CREATE TABLE document_links (
            run_id TEXT NOT NULL, case_uid TEXT NOT NULL, round_index INTEGER,
            doc_id TEXT NOT NULL, section_id TEXT,
            relation TEXT NOT NULL CHECK (relation IN ('resolves', 'supports', 'equivalent', 'hard_negative')),
            grade INTEGER NOT NULL, basis TEXT NOT NULL
        );
        """
    )
    conn.close()
    code, out = _invoke("documents", RUN_ID, "--sqlite-path", str(db_path), "--profile", "docs")
    assert code == 1
    assert "document_links.cause" in out
