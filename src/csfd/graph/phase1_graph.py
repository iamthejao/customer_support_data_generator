"""Phase 1 subgraph — Problem Database generation.

This module defines the LangGraph subgraph that ports
:func:`csfd.pipeline.generate_problem_database` onto LangGraph. Each problem
flows through a generator -> (optional) checker -> commit cycle, with bounded
retries on checker failure. The subgraph shares
:class:`csfd.graph.pipeline_graph.PipelineState` with the parent graph and the
Phase 2 subgraph; no input/output mapping is needed.

Subgraph topology:

    START -> init_phase1 -> generate_problem
                                |
                                v
              _route_after_generate_problem
                /                       \\
        validate_problem           _mark_validation_skipped
              |                                |
        _route_after_validate_problem          |
        /        |          \\                 |
      pass     retry      exhausted            |
        |        |            |                |
        |   _bump_retry   _mark_exhausted      |
        |        |            |                |
        |        v            v                v
        |   generate_problem  commit_problem <-+
        |                          ^
        +--------------------------+
                                   |
                       _route_after_commit_problem
                              /             \\
                            next            done
                              |               \\
                         generate_problem      END
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from csfd.agents.base import AgentContext, Verdict
from csfd.agents.factory import AgentFactory
from csfd.agents.tracing import ParentLink, TracingAdapter
from csfd.graph.pipeline_graph import PipelineState
from csfd.pipeline import ProblemBrainstormOutput, _assign_target_complexities
from csfd.settings import AppSettings
from csfd.storage.db import Database
from csfd.storage.v2_repository import ProblemV2Record, ProblemV2Repo

# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


async def init_phase1_node(
    state: PipelineState,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    """Compute deterministic per-slot target complexities and reset progress."""
    targets = _assign_target_complexities(
        count=settings.problem_database.count,
        proportions=settings.problem_database.complexity_proportions,
    )
    return {
        "target_complexities": [c.value for c in targets],
        "problem_index": 0,
    }


async def generate_problem_node(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
) -> dict[str, Any]:
    """Invoke the problem-brainstorm generator for the current slot."""
    i = state.problem_index
    target_complexity = state.target_complexities[i]
    problem_id = f"{state.run_id}:p:{i:04d}"

    inputs: dict[str, Any] = {
        "company_name": state.company.name,
        "company_overview": state.company.raw_markdown[:2000],
        "scenarios": [dataclasses.asdict(s) for s in state.scenarios.scenarios[:8]],
        "target_complexity": target_complexity,
        "prior_titles": [p.title for p in state.problems_committed[-10:]],
        "index": i,
    }

    link = ParentLink()
    generator = factory.build_generator(
        name="problem_brainstorm",
        prompt_name="phase1.problem_brainstorm_v2",
        output_schema_factory=lambda: ProblemBrainstormOutput,
    )
    traced_gen = TracingAdapter(
        inner=generator,
        db=db,
        run_id=state.run_id,
        node_name="problem_brainstorm",
        artifact_type="problem",
        artifact_id=problem_id,
        role="generator",
        parent_link=link,
    )
    # Feed the prior checker verdict back so the prompt's "prior attempt
    # failed validation" block has actual issues to point at on retry.
    prior_verdicts = [state.last_verdict] if state.last_verdict is not None else []
    result = await traced_gen.invoke(
        AgentContext(
            inputs=inputs, retry_attempt=state.retry_attempt, prior_verdicts=prior_verdicts
        )
    )
    assert isinstance(result, ProblemBrainstormOutput)
    return {
        "current_problem_draft": result,
        "last_generator_trace_id": link.last_generator_trace_id,
    }


async def validate_problem_node(
    state: PipelineState,
    *,
    factory: AgentFactory,
    db: Database,
) -> dict[str, Any]:
    """Run the combined checker against the current problem draft."""
    i = state.problem_index
    problem_id = f"{state.run_id}:p:{i:04d}"
    assert state.current_problem_draft is not None

    inputs: dict[str, Any] = {
        "company_name": state.company.name,
        "target_complexity": state.target_complexities[i],
        "candidate": state.current_problem_draft.model_dump(),
    }

    link = ParentLink(last_generator_trace_id=state.last_generator_trace_id)
    checker = factory.build_checker(
        name="combined_problem_check",
        prompt_name="phase1.problem_combined_check",
    )
    traced_check = TracingAdapter(
        inner=checker,
        db=db,
        run_id=state.run_id,
        node_name="combined_problem_check",
        artifact_type="problem",
        artifact_id=problem_id,
        role="checker",
        parent_link=link,
    )
    verdict = await traced_check.invoke(
        AgentContext(inputs=inputs, retry_attempt=state.retry_attempt, prior_verdicts=[])
    )
    assert isinstance(verdict, Verdict)
    return {
        "last_verdict": verdict,
        "last_quality_flag": None if verdict.passed else state.last_quality_flag,
    }


async def commit_problem_node(
    state: PipelineState,
    *,
    db: Database,
) -> dict[str, Any]:
    """Persist the current draft as a ``ProblemV2Record`` and advance the index."""
    i = state.problem_index
    problem_id = f"{state.run_id}:p:{i:04d}"
    assert state.current_problem_draft is not None
    # Force-fit the LLM's complexity to the target — proportions are authoritative.
    target_complexity = state.target_complexities[i]

    record = ProblemV2Record(
        id=problem_id,
        run_id=state.run_id,
        title=state.current_problem_draft.title,
        summary=state.current_problem_draft.summary,
        background=state.current_problem_draft.background,
        category=state.current_problem_draft.category,
        complexity=target_complexity,
        resolution_hints=state.current_problem_draft.resolution_hints,
        quality_flag=state.last_quality_flag,
        created_at=datetime.now(UTC),
    )
    ProblemV2Repo(db).create(record)

    # ``problems_committed`` has no reducer in PipelineState — return the full
    # list to perform a full-list replacement.
    return {
        "problems_committed": [*state.problems_committed, record],
        "problem_index": state.problem_index + 1,
        "retry_attempt": 0,
        "current_problem_draft": None,
        "last_verdict": None,
        "last_quality_flag": None,
        "last_generator_trace_id": None,
    }


# --------------------------------------------------------------------------- #
# Small helper nodes (retry / quality-flag bookkeeping)
# --------------------------------------------------------------------------- #


async def _bump_retry_node(state: PipelineState) -> dict[str, Any]:
    """Increment the per-artifact retry counter before re-entering generation."""
    return {"retry_attempt": state.retry_attempt + 1}


async def _mark_exhausted_node(state: PipelineState) -> dict[str, Any]:
    """Stamp the retries-exhausted warning before committing a failed draft."""
    return {"last_quality_flag": "warning:retries_exhausted"}


async def _mark_validation_skipped_node(state: PipelineState) -> dict[str, Any]:
    """Stamp the validation-skipped warning when checks are disabled."""
    return {"last_quality_flag": "warning:validation_skipped"}


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #


def _route_after_generate_problem(state: PipelineState) -> Literal["validate", "commit"]:
    return "validate" if state.validation_enabled else "commit"


def _route_after_validate_problem(state: PipelineState) -> Literal["pass", "retry", "exhausted"]:
    assert state.last_verdict is not None, "validate_problem_node must set last_verdict"
    if state.last_verdict.passed:
        return "pass"
    if state.retry_attempt >= state.max_retries:
        return "exhausted"
    return "retry"


def _route_after_commit_problem(state: PipelineState) -> Literal["next", "done"]:
    return "next" if state.problem_index < len(state.target_complexities) else "done"


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def build_phase1_subgraph(
    *,
    factory: AgentFactory,
    db: Database,
    settings: AppSettings,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the Phase 1 (Problem Database) subgraph.

    Returns a graph that, given a ``PipelineState`` with ``company``,
    ``scenarios``, and run-scoped fields populated, generates
    ``settings.problem_database.count`` problems sequentially and persists
    them via :class:`ProblemV2Repo`. No checkpointer is attached here — the
    parent graph owns checkpoint persistence.
    """
    g: StateGraph[PipelineState, Any, PipelineState, PipelineState] = StateGraph(PipelineState)
    g.add_node("init_phase1", partial(init_phase1_node, settings=settings))
    g.add_node("generate_problem", partial(generate_problem_node, factory=factory, db=db))
    g.add_node("validate_problem", partial(validate_problem_node, factory=factory, db=db))
    g.add_node("commit_problem", partial(commit_problem_node, db=db))
    g.add_node("_bump_retry", _bump_retry_node)
    g.add_node("_mark_exhausted", _mark_exhausted_node)
    g.add_node("_mark_validation_skipped", _mark_validation_skipped_node)

    g.add_edge(START, "init_phase1")
    g.add_edge("init_phase1", "generate_problem")
    g.add_conditional_edges(
        "generate_problem",
        _route_after_generate_problem,
        {
            "validate": "validate_problem",
            "commit": "_mark_validation_skipped",
        },
    )
    g.add_edge("_mark_validation_skipped", "commit_problem")
    g.add_conditional_edges(
        "validate_problem",
        _route_after_validate_problem,
        {
            "pass": "commit_problem",
            "retry": "_bump_retry",
            "exhausted": "_mark_exhausted",
        },
    )
    g.add_edge("_bump_retry", "generate_problem")
    g.add_edge("_mark_exhausted", "commit_problem")
    g.add_conditional_edges(
        "commit_problem",
        _route_after_commit_problem,
        {
            "next": "generate_problem",
            "done": END,
        },
    )

    return g.compile()
