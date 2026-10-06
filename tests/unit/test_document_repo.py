"""DocumentRepo: a build is stored whole and replaces the previous one."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from csfd.documents.build import BuildResult, BuiltDocument, CaseAnswer, DocumentLink
from csfd.storage.db import Database
from csfd.storage.documents import DocumentRepo
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import RunRecord, RunRepo
from tests.fixtures.sample_documents import figure_asset, kb_article, service_manual

RUN = "run-docs"


def _db(path: Path) -> Database:
    db = Database(path=path)
    apply_migrations(db)
    now = datetime.now(UTC)
    RunRepo(db).create(
        RunRecord(
            id=RUN,
            phase="full",
            parent_run_id=None,
            status="completed",
            started_at=now,
            completed_at=now,
            run_seed=1,
            pipeline_version="test",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )
    return db


def _result() -> BuildResult:
    manual = service_manual("Acme")
    article = kb_article("Acme", "run:p:0000", 0)
    return BuildResult(
        documents=[
            BuiltDocument(document=article),
            BuiltDocument(document=manual, assets=[figure_asset()]),
        ],
        links=[
            DocumentLink(
                case_uid="c1",
                doc_id=article.doc_id,
                section_id="sec-2",
                relation="resolves",
                grade=2,
                basis="root_cause_fix",
            ),
            DocumentLink(
                case_uid="c1",
                round_index=2,
                doc_id=manual.doc_id,
                section_id=None,
                relation="supports",
                grade=1,
                basis="requested_document",
            ),
        ],
        cases=[CaseAnswer(case_uid="c1", answer_source="partial")],
    )


def test_round_trip(tmp_db_path: Path) -> None:
    db = _db(tmp_db_path)
    repo = DocumentRepo(db)
    assert not repo.has_documents(RUN)
    result = _result()
    made_by = {b.document.doc_id: "sample" for b in result.documents}
    repo.replace_for_run(RUN, result, made_by)
    assert repo.has_documents(RUN)
    docs = repo.list_for_run(RUN)
    # Library tier first, then problem tier.
    assert [d.doc_number for d in docs] == ["KD-SM-CF600-EN", "KB-0000"]
    assert docs[0] == result.documents[1].document
    assets = repo.assets_for_run(RUN)
    assert assets == {figure_asset().asset_id: figure_asset()}
    assert repo.links_for_run(RUN) == sorted(
        result.links, key=lambda link: (link.case_uid, link.round_index or 0, link.doc_id)
    )
    assert repo.answer_sources(RUN) == {"c1": "partial"}


def test_a_new_build_replaces_the_old_one(tmp_db_path: Path) -> None:
    db = _db(tmp_db_path)
    repo = DocumentRepo(db)
    first = _result()
    repo.replace_for_run(RUN, first, {b.document.doc_id: "sample" for b in first.documents})
    article = kb_article("Acme", "run:p:0001", 1)
    repo.replace_for_run(
        RUN, BuildResult(documents=[BuiltDocument(document=article)]), {article.doc_id: "sample"}
    )
    assert [d.doc_id for d in repo.list_for_run(RUN)] == [article.doc_id]
    assert repo.assets_for_run(RUN) == {}
    assert repo.links_for_run(RUN) == []
    assert repo.answer_sources(RUN) == {}
