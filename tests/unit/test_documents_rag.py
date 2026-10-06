"""BEIR-style retrieval files: corpus, queries, graded and resolves-only qrels."""

from __future__ import annotations

import json
from pathlib import Path

from csfd.documents.build import DocumentLink
from csfd.documents.rag import Query, write_rag_files
from csfd.documents.sidecar import chunks
from tests.fixtures.sample_documents import kb_article, service_manual

MANUAL = service_manual("Acme")
ARTICLE = kb_article("Acme", "run:p:0000", 0)
DOCS = [(MANUAL, chunks(MANUAL, {})), (ARTICLE, chunks(ARTICLE, {}))]
QUERIES = [
    Query(
        case_uid="run:000000", problem_id="run:p:0000", ticket_type="l1", subject="s0", text="t0"
    ),
    Query(
        case_uid="run:000001", problem_id="run:p:0000", ticket_type="l2", subject="s1", text="t1"
    ),
]


def _link(**kw: object) -> DocumentLink:
    fields: dict[str, object] = {"case_uid": "run:000000", "doc_id": ARTICLE.doc_id, "basis": "x"}
    fields.update(kw)
    return DocumentLink.model_validate(fields)


LINKS = [
    _link(section_id="sec-2", relation="resolves", grade=2),
    _link(section_id="sec-1", relation="supports", grade=1),
    _link(section_id="sec-1", relation="supports", grade=1, round_index=2),
    _link(section_id="sec-3", relation="hard_negative", grade=0),
    # A whole-document link applies to every section of the document.
    _link(doc_id=MANUAL.doc_id, section_id=None, relation="supports", grade=1),
]


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tsv(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]


def test_corpus_has_one_row_per_section(tmp_path: Path) -> None:
    write_rag_files(tmp_path, DOCS, QUERIES, LINKS, {})
    corpus = _rows(tmp_path / "rag" / "corpus.jsonl")
    assert len(corpus) == 5 + 3
    silver = next(r for r in corpus if r["_id"] == f"{MANUAL.doc_id}#sec-6.2")
    assert silver["title"] == "CF-600 Ceramic Firing Furnace > 6 Calibration > 6.2 Silver test"
    assert silver["metadata"]["doc_number"] == "KD-SM-CF600-EN"  # type: ignore[index]


def test_queries_never_collide_with_corpus_ids_and_say_if_documents_answer(
    tmp_path: Path,
) -> None:
    write_rag_files(tmp_path, DOCS, QUERIES, LINKS, {"run:000001": "agent_knowledge"})
    queries = _rows(tmp_path / "rag" / "queries.jsonl")
    assert [q["_id"] for q in queries] == ["q:run:000000", "q:run:000001"]
    meta = [q["metadata"] for q in queries]
    assert meta[0]["answerable_from_documents"] is True  # type: ignore[index]
    assert meta[1]["answerable_from_documents"] is False  # type: ignore[index]
    assert meta[1]["answer_source"] == "agent_knowledge"  # type: ignore[index]
    assert meta[0]["answer_source"] is None  # type: ignore[index]


def test_qrels_are_integer_grades_with_hard_negatives_left_out(tmp_path: Path) -> None:
    write_rag_files(tmp_path, DOCS, QUERIES, LINKS, {})
    graded = _tsv(tmp_path / "rag" / "qrels" / "test.tsv")
    assert graded[0] == ["query-id", "corpus-id", "score"]
    pairs = {(q, c): int(s) for q, c, s in graded[1:]}
    assert pairs[("q:run:000000", f"{ARTICLE.doc_id}#sec-2")] == 2
    assert pairs[("q:run:000000", f"{ARTICLE.doc_id}#sec-1")] == 1  # two contacts, one row
    assert ("q:run:000000", f"{ARTICLE.doc_id}#sec-3") not in pairs
    assert sum(1 for q, c in pairs if c.startswith(MANUAL.doc_id)) == 5
    resolves = _tsv(tmp_path / "rag" / "qrels" / "test.resolves.tsv")
    assert resolves[1:] == [["q:run:000000", f"{ARTICLE.doc_id}#sec-2", "1"]]


def test_detailed_qrels_keep_relation_basis_contact_and_negatives(tmp_path: Path) -> None:
    write_rag_files(tmp_path, DOCS, QUERIES, LINKS, {})
    detailed = _rows(tmp_path / "rag" / "qrels_detailed.jsonl")
    negative = next(r for r in detailed if r["relation"] == "hard_negative")
    assert negative["score"] == 0
    assert any(r["round_index"] == 2 for r in detailed)
    assert len(detailed) == 4 + 5
