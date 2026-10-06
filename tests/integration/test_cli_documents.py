"""CLI: `csfd documents` (back-fill) and `csfd export --format documents`, end to end.

Uses the fixture builder (tests.fixtures.sample_documents) registered for the
test; no builder ships yet and no model is called.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from csfd.cli import app
from csfd.documents import build
from csfd.storage.db import Database
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
from tests.fixtures.sample_documents import SampleBuilder

runner = CliRunner()
RUN_ID = "run-docs"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A checkout-like folder: default config plus a profile that enables documents."""
    config = tmp_path / "config"
    (config / "profiles").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "config" / "default.yaml", config / "default.yaml")
    (config / "profiles" / "docs.yaml").write_text(
        "documents:\n  enabled: true\n  builders: [sample]\n", encoding="utf-8"
    )
    (config / "profiles" / "nobuilders.yaml").write_text(
        "documents:\n  enabled: true\n", encoding="utf-8"
    )
    (config / "profiles" / "typo.yaml").write_text(
        "documents:\n  enabled: true\n  builders: [smaple]\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(build.BUILDERS, "sample", SampleBuilder)
    return tmp_path


def _seed(db_path: Path, run_id: str = RUN_ID, *, case: bool = True) -> None:
    db = Database(path=db_path)
    apply_migrations(db)
    now = datetime.now(UTC)
    RunRepo(db).create(
        RunRecord(
            id=run_id,
            phase="full",
            parent_run_id=None,
            status="completed",
            started_at=now,
            completed_at=now,
            run_seed=1,
            pipeline_version="test",
            git_sha=None,
            config_snapshot_json=json.dumps(
                {"company": {"name": "Kalvora Dental", "seed": "kalvora"}}
            ),
            stats_json=None,
            error_summary=None,
        )
    )
    problem_id = f"{run_id}:p:0000"
    ProblemRepo(db).create(
        ProblemRecord(
            id=problem_id,
            run_id=run_id,
            title="Over-firing",
            summary="s",
            background="b",
            category="firing",
            complexity="simple",
            resolution_hints={},
            quality_flag=None,
            created_at=now,
        )
    )
    if not case:
        return
    ticket_uid = f"{run_id}:000000"
    LineageRepo(db).create(
        LineageRecord(
            ticket_uid=ticket_uid,
            run_id=run_id,
            slot_index=0,
            problem_id=problem_id,
            ticket_type="l1",
            customer_tier="standard",
            customer_tone="neutral",
            incoming_request_id=None,
            resolution_id=None,
            created_at=now,
            case_facts={
                "asset_model": "CF-600",
                "asset_serial": "CF600-1234-AB",
                "customer_company": "Lab",
                "site_city": "Bern",
                "site_country": "Switzerland",
            },
        )
    )
    ir_id = IncomingRequestRepo(db).create(
        IncomingRequestRecord(
            request_uid=f"{ticket_uid}:req",
            run_id=run_id,
            problem_id=problem_id,
            ticket_type="l1",
            customer_name="Customer-standard-0000",
            customer_tier="standard",
            customer_tone="neutral",
            channel="email",
            subject="Crowns come out glossy",
            body="Since last week our CF-600 over-fires everything.",
            quality_flag=None,
            created_at=now,
            case_uid=ticket_uid,
        )
    )
    LineageRepo(db).update_links(ticket_uid, incoming_request_id=ir_id)


def _invoke(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def _tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_documents_are_off_by_default(workdir: Path) -> None:
    db = workdir / "runs.sqlite"
    _seed(db)
    code, out = _invoke("documents", RUN_ID, "--sqlite-path", str(db))
    assert code == 1
    assert "documents.enabled: false" in out


def test_no_builders_configured_builds_nothing(workdir: Path) -> None:
    db = workdir / "runs.sqlite"
    _seed(db)
    code, out = _invoke("documents", RUN_ID, "--sqlite-path", str(db), "--profile", "nobuilders")
    assert code == 0, out
    assert "No document builders are configured" in out


def test_unknown_builder_and_unknown_run_are_errors(workdir: Path) -> None:
    db = workdir / "runs.sqlite"
    _seed(db)
    code, out = _invoke("documents", RUN_ID, "--sqlite-path", str(db), "--profile", "typo")
    assert code == 2
    assert "smaple" in out
    code, out = _invoke("documents", "nope", "--sqlite-path", str(db), "--profile", "docs")
    assert code == 2
    assert "no run 'nope'" in out


def test_build_then_export_writes_documents_sidecars_and_rag_files(workdir: Path) -> None:
    db = workdir / "runs.sqlite"
    _seed(db)
    code, out = _invoke("documents", RUN_ID, "--sqlite-path", str(db), "--profile", "docs")
    assert code == 0, out
    assert "Built 3 documents and 5 links" in out

    exports = workdir / "exports"
    code, out = _invoke(
        "export", RUN_ID, "--format", "documents", "--sqlite-path", str(db), "--out", str(exports)
    )
    assert code == 0, out
    root = exports / RUN_ID / "documents"
    manual = root / "docs" / "KD-SM-CF600-EN_revC"
    for name in [
        "KD-SM-CF600-EN_revC.docx",
        "KD-SM-CF600-EN_revC.pdf",
        "KD-SM-CF600-EN_revC.md",
        "KD-SM-CF600-EN_revC.json",
        "assets/fig-3-1.svg",
        "assets/fig-3-1.png",
    ]:
        assert (manual / name).is_file(), name
    assert (root / "docs" / "KD-SM-CF600-EN_revB").is_dir()  # the superseded revision
    assert (root / "docs" / "KB-0000_revA" / "KB-0000_revA.pdf").is_file()

    index = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines()]
    assert [r["doc_number"] for r in index] == ["KD-SM-CF600-EN", "KD-SM-CF600-EN", "KB-0000"]
    side = json.loads((manual / "KD-SM-CF600-EN_revC.json").read_text())
    assert (
        side["files"]["KD-SM-CF600-EN_revC.pdf"]
        == hashlib.sha256((manual / "KD-SM-CF600-EN_revC.pdf").read_bytes()).hexdigest()
    )
    assert all(c["pdf_pages"] for c in side["chunks"])

    queries = [
        json.loads(line) for line in (root / "rag" / "queries.jsonl").read_text().splitlines()
    ]
    assert queries == [
        {
            "_id": f"q:{RUN_ID}:000000",
            "text": "Since last week our CF-600 over-fires everything.",
            "metadata": {
                "case_uid": f"{RUN_ID}:000000",
                "problem_id": f"{RUN_ID}:p:0000",
                "ticket_type": "l1",
                "subject": "Crowns come out glossy",
                "answer_source": "documents",
                "answerable_from_documents": True,
            },
        }
    ]
    resolves = (root / "rag" / "qrels" / "test.resolves.tsv").read_text().splitlines()
    assert resolves[1].endswith("KB-0000:en:rev-A#sec-2\t1")

    manifest = json.loads((root / "manifest.json").read_text())
    tree = _tree(root)
    tree.pop("manifest.json")
    assert manifest["files"] == tree

    # Exporting again gives the same bytes (stale files of the first export are cleared).
    (root / "stale.txt").write_text("old")
    code, out = _invoke(
        "export", RUN_ID, "--format", "documents", "--sqlite-path", str(db), "--out", str(exports)
    )
    assert code == 0, out
    again = _tree(root)
    again.pop("manifest.json")
    assert again == tree


def test_rebuilding_replaces_the_previous_build(workdir: Path) -> None:
    db = workdir / "runs.sqlite"
    _seed(db)
    for _ in range(2):
        code, out = _invoke("documents", RUN_ID, "--sqlite-path", str(db), "--profile", "docs")
        assert code == 0, out
        assert "Built 3 documents and 5 links" in out


def test_export_all_writes_no_documents_for_a_run_without_them(workdir: Path) -> None:
    db = workdir / "runs.sqlite"
    _seed(db, case=False)
    exports = workdir / "exports"
    code, out = _invoke(
        "export", RUN_ID, "--format", "all", "--sqlite-path", str(db), "--out", str(exports)
    )
    assert code == 0, out
    assert not (exports / RUN_ID / "documents").exists()
    code, out = _invoke(
        "export", RUN_ID, "--format", "documents", "--sqlite-path", str(db), "--out", str(exports)
    )
    assert code == 0, out
    assert "has no documents" in out


def test_a_database_from_before_documents_still_exports_and_can_be_back_filled(
    workdir: Path,
) -> None:
    db_path = workdir / "runs.sqlite"
    _seed(db_path)
    db = Database(path=db_path)
    with db.connect() as conn:
        for table in ("document_links", "document_cases", "document_assets", "documents"):
            conn.execute(f"DROP TABLE {table}")
    exports = workdir / "exports"
    code, out = _invoke(
        "export", RUN_ID, "--format", "all", "--sqlite-path", str(db_path), "--out", str(exports)
    )
    assert code == 0, out
    assert not (exports / RUN_ID / "documents").exists()
    code, out = _invoke("documents", RUN_ID, "--sqlite-path", str(db_path), "--profile", "docs")
    assert code == 0, out
    assert "Built 3 documents" in out
