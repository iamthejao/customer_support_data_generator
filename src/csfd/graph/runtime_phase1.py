"""Runtime-wired Phase 1 LangGraph — single-problem subgraph that runs end-to-end.

This module composes the Phase 1 KB-generation nodes (defined in
``csfd.phases.phase1_kb.nodes``) into a fully runnable LangGraph by binding
the ``AgentFactory`` and ``Database`` dependencies via ``functools.partial``.

The graph processes ONE problem end-to-end:
    brainstorm -> 3 parallel checkers -> aggregate -> route -> commit (or
    regenerate) -> coverage decision (+ persist problem) -> optional article
    write -> persist article.

The outer multi-problem loop is implemented separately in ``compose.py``.
"""

from __future__ import annotations

from functools import partial
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from csfd.agents.factory import AgentFactory
from csfd.phases.phase1_kb.nodes import (
    article_writer_node,
    coverage_decider_node,
    persist_article_node,
    persist_problem_node,
    problem_background_check_node,
    problem_brainstorm_node,
    problem_consistency_check_node,
    problem_scenario_check_node,
)
from csfd.phases.phase1_kb.routing import (
    dispatch_problem_checkers,
    route_problem_verdict,
)
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBState,
)
from csfd.storage.db import Database


def _route_problem(state: KBState, *, max_retries: int) -> str:
    """Route looking only at the most recent 3 verdicts.

    Verdicts accumulate across retries (operator.add reducer), so we slice
    the last three (one per checker for the most-recent attempt) before
    delegating to the underlying ``route_problem_verdict``.
    """
    last3 = state.verdicts[-3:] if len(state.verdicts) >= 3 else state.verdicts
    snapshot = state.model_copy(update={"verdicts": last3})
    return route_problem_verdict(snapshot, max_retries=max_retries)


def _bump_retry(state: KBState) -> dict[str, Any]:
    """Increment the retry counter before looping back to brainstorm."""
    return {"retry_attempt": state.retry_attempt + 1}


def _aggregate_verdicts_node(state: KBState) -> dict[str, Any]:
    """Fan-in barrier after the 3 problem checkers.

    LangGraph requires nodes to write at least one state-schema field, so
    we re-write ``current_problem_draft`` (a regular non-reducer field) as
    a no-op state update.
    """
    return {"current_problem_draft": state.current_problem_draft}


async def _commit_problem_node(state: KBState) -> dict[str, Any]:
    """Move the current draft into the ``problems_committed`` list.

    Subsequent nodes (coverage, persist, article write) operate on the
    last entry of that list.
    """
    if state.current_problem_draft is None:
        raise ValueError("commit_problem_node called without current_problem_draft")
    pc = CommittedProblem(
        id=str(uuid4()),
        draft=state.current_problem_draft,
        has_kb=False,
        coverage_reasoning="",
        coverage_confidence="low",
    )
    return {"problems_committed": [pc]}


async def _coverage_then_persist(
    state: KBState,
    *,
    factory: AgentFactory,
    db: Database,
    target_rate: float,
) -> dict[str, Any]:
    """Run the coverage decider and persist the committed problem.

    The coverage decider mutates the committed problem with the judgement
    (``has_kb`` + reasoning/confidence). After applying the update we persist
    the (now fully-populated) problem to SQLite.
    """
    if not state.problems_committed:
        return {}
    update = await coverage_decider_node(state, factory=factory, target_rate=target_rate)
    snapshot = state.model_copy(update=update)
    pc = snapshot.problems_committed[-1]
    persist_problem_node(snapshot, db=db, problem=pc)
    return {"problems_committed": snapshot.problems_committed}


async def _write_article_if_covered(
    state: KBState,
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    """Write a KB article for the last committed problem when it has coverage."""
    if not state.problems_committed:
        return {}
    pc = state.problems_committed[-1]
    if not pc.has_kb:
        return {}
    return await article_writer_node(state, factory=factory, problem=pc)


async def _persist_article(state: KBState, *, db: Database) -> dict[str, Any]:
    """Persist the current article draft to SQLite, if any.

    Always returns at least one state-schema field so LangGraph's
    require-at-least-one-write check is satisfied even on the no-op
    branches (no committed problem / coverage = False / no draft).
    """
    if state.current_article_draft is None or not state.problems_committed:
        return {"current_article_draft": state.current_article_draft}
    pc = state.problems_committed[-1]
    if not pc.has_kb:
        return {"current_article_draft": state.current_article_draft}
    article = CommittedArticle(
        id=str(uuid4()),
        problem_id=pc.id,
        draft=state.current_article_draft,
    )
    return persist_article_node(state, db=db, article=article)


def _route_after_coverage(state: KBState) -> str:
    """Choose whether to write a KB article after coverage decision."""
    pc = state.problems_committed[-1] if state.problems_committed else None
    return "write_article" if pc is not None and pc.has_kb else "done"


def build_runtime_phase1_graph(
    *,
    factory: AgentFactory,
    db: Database,
    max_retries: int,
    kb_target_rate: float,
) -> CompiledStateGraph:
    """Build (and compile) a runtime Phase 1 graph for a single problem.

    Args:
        factory: AgentFactory producing all Phase 1 agents (brainstorm, 3
            problem checkers, coverage judge, article writer).
        db: SQLite ``Database`` for persisting problems and articles.
        max_retries: Maximum brainstorm retries before forcibly committing
            a problem with a quality warning.
        kb_target_rate: Fraction of committed problems that should have a
            KB article (used by the coverage decider's top-up/trim logic).

    Returns:
        A compiled LangGraph ready to be ``ainvoke``d with a ``KBState``.
    """
    g: StateGraph = StateGraph(KBState)

    g.add_node(
        "problem_brainstorm",
        partial(problem_brainstorm_node, factory=factory),
    )
    g.add_node(
        "problem_consistency_check",
        partial(problem_consistency_check_node, factory=factory),
    )
    g.add_node(
        "problem_background_check",
        partial(problem_background_check_node, factory=factory),
    )
    g.add_node(
        "problem_scenario_check",
        partial(problem_scenario_check_node, factory=factory),
    )
    g.add_node("problem_aggregate", _aggregate_verdicts_node)
    g.add_node("bump_retry", _bump_retry)
    g.add_node("commit_problem", _commit_problem_node)
    g.add_node(
        "coverage_persist",
        partial(
            _coverage_then_persist,
            factory=factory,
            db=db,
            target_rate=kb_target_rate,
        ),
    )
    g.add_node(
        "write_article",
        partial(_write_article_if_covered, factory=factory),
    )
    g.add_node("persist_article", partial(_persist_article, db=db))

    g.add_edge(START, "problem_brainstorm")

    g.add_conditional_edges(
        "problem_brainstorm",
        dispatch_problem_checkers,  # type: ignore[arg-type]
        {
            "problem_consistency_check": "problem_consistency_check",
            "problem_background_check": "problem_background_check",
            "problem_scenario_check": "problem_scenario_check",
        },
    )
    g.add_edge("problem_consistency_check", "problem_aggregate")
    g.add_edge("problem_background_check", "problem_aggregate")
    g.add_edge("problem_scenario_check", "problem_aggregate")

    def _route(state: KBState) -> str:
        return _route_problem(state, max_retries=max_retries)

    g.add_conditional_edges(
        "problem_aggregate",
        _route,
        {
            "commit": "commit_problem",
            "commit_with_warning": "commit_problem",
            "regenerate": "bump_retry",
        },
    )
    g.add_edge("bump_retry", "problem_brainstorm")
    g.add_edge("commit_problem", "coverage_persist")
    g.add_conditional_edges(
        "coverage_persist",
        _route_after_coverage,
        {"write_article": "write_article", "done": END},
    )
    g.add_edge("write_article", "persist_article")
    g.add_edge("persist_article", END)

    return g.compile()
