"""Retrieval-evaluation files in the BEIR / MTEB layout.

* ``corpus.jsonl`` — one row per document section: ``{"_id", "title", "text", "metadata"}``.
* ``queries.jsonl`` — one row per case: the customer's opening message (what a
  system under test receives), never the agent's turns, which may name the answer.
* ``qrels/test.tsv`` — ``query-id  corpus-id  score`` with integer grades
  2 (resolves) and 1 (supports); a header row first.
* ``qrels/test.resolves.tsv`` — only the grade-2 pairs, scored 1, for "did it find
  the fix" metrics (BEIR's evaluator counts grade 1 as relevant too).
* ``qrels_detailed.jsonl`` — every link with its relation, basis and contact,
  including hard negatives (grade 0).

Query ids are ``q:<case_uid>`` so they never collide with corpus ids. A link to a
whole document (no section) applies to every section of that document.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from csfd.documents.build import AnswerSource, DocumentLink
from csfd.documents.ir import DocumentIR
from csfd.documents.sidecar import Chunk, chunk_id


class Query(BaseModel):
    case_uid: str
    problem_id: str
    ticket_type: str
    subject: str
    text: str


def query_id(case_uid: str) -> str:
    return f"q:{case_uid}"


def _jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return path


def _tsv(path: Path, rows: Sequence[tuple[str, str, int]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["query-id\tcorpus-id\tscore", *(f"{q}\t{c}\t{s}" for q, c, s in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_rag_files(
    out_dir: Path,
    docs: Sequence[tuple[DocumentIR, list[Chunk]]],
    queries: Sequence[Query],
    links: Sequence[DocumentLink],
    answer_sources: Mapping[str, AnswerSource],
) -> list[Path]:
    """Write the retrieval files under ``out_dir``; returns the paths written."""
    corpus = []
    sections_of: dict[str, list[str]] = {}
    for doc, doc_chunks in docs:
        sections_of[doc.doc_id] = [c.section_id for c in doc_chunks]
        for c in doc_chunks:
            corpus.append(
                {
                    "_id": c.chunk_id,
                    "title": " > ".join([doc.title, *c.heading_path]),
                    "text": c.text,
                    "metadata": {
                        "doc_id": doc.doc_id,
                        "section_id": c.section_id,
                        "doc_number": doc.doc_number,
                        "revision": doc.revision,
                        "status": doc.status,
                        "doc_type": doc.doc_type,
                        "tier": doc.tier,
                        "audience": doc.audience,
                        "language": doc.language,
                        "models": doc.applies_to.models,
                    },
                }
            )

    detailed = []
    best: dict[tuple[str, str], int] = {}
    for link in links:
        targets = [link.section_id] if link.section_id else sections_of.get(link.doc_id, [])
        for section_id in targets:
            qid, cid = query_id(link.case_uid), chunk_id(link.doc_id, section_id)
            detailed.append(
                {
                    "query-id": qid,
                    "corpus-id": cid,
                    "score": link.grade,
                    "relation": link.relation,
                    "basis": link.basis,
                    "round_index": link.round_index,
                }
            )
            if link.grade > 0:
                best[(qid, cid)] = max(best.get((qid, cid), 0), link.grade)

    resolving = {q for (q, _), grade in best.items() if grade == 2}
    query_rows = [
        {
            "_id": query_id(q.case_uid),
            "text": q.text,
            "metadata": {
                "case_uid": q.case_uid,
                "problem_id": q.problem_id,
                "ticket_type": q.ticket_type,
                "subject": q.subject,
                "answer_source": answer_sources.get(q.case_uid),
                "answerable_from_documents": query_id(q.case_uid) in resolving,
            },
        }
        for q in queries
    ]
    graded = sorted((q, c, g) for (q, c), g in best.items())
    rag = out_dir / "rag"
    return [
        _jsonl(rag / "corpus.jsonl", corpus),
        _jsonl(rag / "queries.jsonl", query_rows),
        _tsv(rag / "qrels" / "test.tsv", graded),
        _tsv(rag / "qrels" / "test.resolves.tsv", [(q, c, 1) for q, c, g in graded if g == 2]),
        _jsonl(rag / "qrels_detailed.jsonl", detailed),
    ]
