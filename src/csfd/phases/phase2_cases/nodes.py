"""Async LangGraph nodes for the Phase 2 case-generation subgraph."""

from __future__ import annotations

import json
from typing import Any

from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBArticleDraft,
    ProblemDraft,
)
from csfd.phases.phase2_cases.state import TicketState
from csfd.storage.db import Database
from csfd.storage.repository import KBArticleRepo, ProblemRepo


async def load_kb_node(state: TicketState, *, db: Database) -> dict[str, Any]:
    """Load problems + KB articles from the parent Phase 1 run."""
    problems = ProblemRepo(db).list_for_run(state.parent_run_id)
    pool: list[CommittedProblem] = []
    for p in problems:
        pool.append(
            CommittedProblem(
                id=p.id,
                draft=ProblemDraft(
                    title=p.title,
                    description=p.description,
                    category=p.category,
                    severity=p.severity,
                    metadata=(json.loads(p.metadata_json) if p.metadata_json else {}),
                ),
                has_kb=p.has_kb,
                coverage_reasoning=p.coverage_reasoning or "",
                coverage_confidence=p.coverage_confidence or "low",
                quality_flag=p.quality_flag,
            )
        )

    kb_by_problem: dict[str, CommittedArticle] = {}
    repo = KBArticleRepo(db)
    for cp in pool:
        if not cp.has_kb:
            continue
        row = repo.get_by_problem(cp.id)
        if row is None:
            continue
        kb_by_problem[cp.id] = CommittedArticle(
            id=row.id,
            problem_id=row.problem_id,
            draft=KBArticleDraft(
                title=row.title,
                content_markdown=row.content_markdown,
                troubleshooting_steps=(
                    json.loads(row.troubleshooting_steps_json)
                    if row.troubleshooting_steps_json
                    else []
                ),
                prerequisites=(
                    json.loads(row.prerequisites_json) if row.prerequisites_json else []
                ),
                metadata=(json.loads(row.metadata_json) if row.metadata_json else {}),
            ),
            quality_flag=row.quality_flag,
        )

    return {"problem_pool": pool, "kb_by_problem": kb_by_problem}
