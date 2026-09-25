"""Top-level entry-point for the deterministic CSFD pipeline.

The pipeline runs as a LangGraph graph composed of two phase subgraphs (see
``csfd.graph.pipeline_graph``). This module hosts the entry-point wrapper that
opens the checkpointer and ``ainvoke``s the parent graph, plus the LLM-output
contracts and small pure helpers that are reused by graph nodes:

* :class:`ProblemBrainstormOutput` / :class:`ResolutionOutput` /
  :class:`DialogueTurnOutput` — Pydantic schemas the LLM is asked to produce.
* :func:`_assign_target_complexities` — proportion → integer-count assignment
  via largest-remainder rounding (used by Phase 1's ``init_phase1`` node).
* :func:`_compute_run_stats` — end-of-run aggregation written to
  ``runs.stats_json`` (used by the parent graph's ``finalize_run`` node).
* :func:`run_pipeline` — CLI entry-point; opens an async SQLite checkpointer,
  builds the parent graph, and ``ainvoke``s it once per run.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from csfd.agents.factory import AgentFactory
from csfd.case_plan import CasePlan
from csfd.diagnosis import DiagnosisPlan
from csfd.outcomes import (
    Commitment,
    ContactEnding,
    ProblemState,
    is_resolved,
    observed_ending,
)
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.settings import AppSettings
from csfd.storage.db import Database
from csfd.storage.exporters import run_scope
from csfd.storage.repository import ProblemRepo
from csfd.ticket_types.definitions import CustomerImpact, FaultDomain, ProblemComplexity
from csfd.utils.rng import largest_remainder

if TYPE_CHECKING:
    from csfd.storage.db_async import AsyncDatabase

_log = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# LLM-output schemas
# --------------------------------------------------------------------------- #


class ProblemBrainstormOutput(BaseModel):
    """Schema for the problem-brainstorm generator's structured output."""

    title: str
    summary: str
    background: str
    symptoms: list[str] = Field(default_factory=list)
    root_cause: list[str] = Field(default_factory=list)
    category: str
    complexity: ProblemComplexity
    fault_domain: FaultDomain = FaultDomain.SOFTWARE
    customer_impact: CustomerImpact = CustomerImpact.DEGRADED
    tags: list[str] = Field(default_factory=list)
    resolution_hints: dict[str, str] = Field(default_factory=dict)
    # The canonical diagnosis (candidate causes with their fixes, checks with their
    # findings; see csfd.diagnosis) and the case outcomes this problem can plausibly end in.
    diagnosis_plan: DiagnosisPlan = Field(default_factory=DiagnosisPlan)
    viable_outcomes: list[ProblemState] = Field(default_factory=list)


class DialogueTurnOutput(BaseModel):
    speaker: Literal["customer", "agent"]
    content: str
    done: bool = False
    done_reason: str | None = None
    # Promises made in this turn, recorded by the speaker who made them. Never
    # rendered into the transcript; the consistency check verifies them against it.
    commitments: list[Commitment] = Field(default_factory=list)


class IncomingRequestOutput(BaseModel):
    subject: str
    body: str


# How the dialogue loop ended. "dropped" is a planned mid-conversation cut-off
# of a non-final contact in a multi-contact case (see csfd.rounds).
EndReason = Literal["customer_done", "agent_done", "cap_hit", "dropped"]


class ResolutionOutput(BaseModel):
    subject: str
    body: str
    turns: list[DialogueTurnOutput] = Field(default_factory=list)
    end_reason: EndReason = "agent_done"
    # The state the contact leaves the problem in (see csfd.outcomes): the planned
    # state until the consistency check reports what the dialogue actually reached.
    problem_state: ProblemState | None = None
    contact_ending: ContactEnding | None = None

    @property
    def resolved(self) -> bool:
        """Derived: the problem is fixed (verified or not)."""
        return is_resolved(self.problem_state)

    @property
    def commitments(self) -> list[Commitment]:
        return [c for t in self.turns for c in t.commitments]


class EditedTurn(BaseModel):
    """A turn as the consistency check returns it; omitted `commitments` keep the original's."""

    speaker: Literal["customer", "agent"]
    content: str
    done: bool = False
    done_reason: str | None = None
    commitments: list[Commitment] | None = None


class ConsistencyVerdict(BaseModel):
    status: Literal["pass", "pass_with_edits", "fail"]
    issues: list[str] = Field(default_factory=list)
    edited_turns: list[EditedTurn] | None = None
    edited_subject: str | None = None
    edited_body: str | None = None
    # The state the conversation actually leaves the problem in, judged from the text.
    problem_state: ProblemState | None = None


def _assemble_resolution(
    *,
    subject: str,
    body: str,
    turns: list[DialogueTurnOutput],
    end_reason: EndReason,
    problem_state: ProblemState | None = None,
) -> ResolutionOutput:
    """Build a ResolutionOutput from accumulated dialogue state, deriving the contact ending."""
    last_reason = turns[-1].done_reason if turns else None
    return ResolutionOutput(
        subject=subject,
        body=body,
        turns=turns,
        end_reason=end_reason,
        problem_state=problem_state,
        contact_ending=observed_ending(end_reason, last_reason),
    )


def _carry_commitments(
    original: list[DialogueTurnOutput], edited: list[EditedTurn]
) -> list[DialogueTurnOutput]:
    """Keep each turn's recorded commitments when an edit omits them; an explicit list wins."""
    out: list[DialogueTurnOutput] = []
    for i, turn in enumerate(edited):
        commitments = turn.commitments
        if commitments is None:
            before = original[i] if i < len(original) else None
            same = before is not None and before.speaker == turn.speaker
            commitments = before.commitments if before is not None and same else []
        out.append(
            DialogueTurnOutput(**turn.model_dump(exclude={"commitments"}), commitments=commitments)
        )
    return out


def _apply_consistency_edits(
    draft: ResolutionOutput, verdict: ConsistencyVerdict
) -> ResolutionOutput:
    """Return the draft with the verdict's edits and reported problem state applied.

    Only fields the verdict explicitly sets are replaced; a verdict without a
    problem state keeps the draft's. A natural ending (`customer_done` /
    `agent_done`) follows whoever speaks last in the edited turns, since the
    checker may append a closing turn; `cap_hit` / `dropped` are kept as-is.
    """
    state = verdict.problem_state or draft.problem_state
    if verdict.status != "pass_with_edits":
        return (
            draft
            if state == draft.problem_state
            else draft.model_copy(update={"problem_state": state})
        )
    subject = verdict.edited_subject if verdict.edited_subject is not None else draft.subject
    body = verdict.edited_body if verdict.edited_body is not None else draft.body
    turns = (
        _carry_commitments(draft.turns, verdict.edited_turns)
        if verdict.edited_turns is not None
        else draft.turns
    )
    end_reason = draft.end_reason
    if end_reason in ("customer_done", "agent_done") and turns:
        end_reason = "customer_done" if turns[-1].speaker == "customer" else "agent_done"
    return _assemble_resolution(
        subject=subject, body=body, turns=turns, end_reason=end_reason, problem_state=state
    )


# --------------------------------------------------------------------------- #
# Pure helpers reused by graph nodes
# --------------------------------------------------------------------------- #


def _assign_target_complexities(
    *, count: int, proportions: Mapping[str, float]
) -> list[ProblemComplexity]:
    """Deterministically assign target complexities to `count` problem slots."""
    counts = largest_remainder(proportions, count)
    out: list[ProblemComplexity] = []
    for c, n in counts.items():
        out.extend([ProblemComplexity(c)] * n)
    return out


def _compute_run_stats(
    *,
    db: Database,
    run_id: str,
    started_at: datetime,
) -> dict[str, Any]:
    """Aggregate end-of-run statistics from the persisted tables.

    Returns counts plus a quality-flag breakdown so reviewers can read run
    health without joining trace rows. Stored in ``runs.stats_json``.
    """

    def _group_count(
        table: str, group_col: str, extra_where: str = "", none_label: str = "ok"
    ) -> dict[str, int]:
        with db.connect() as conn:
            rows = conn.execute(
                f"SELECT {group_col} AS k, COUNT(*) AS n FROM {table} "
                f"WHERE {run_scope(table)} {extra_where} GROUP BY {group_col} ORDER BY {group_col}",
                (run_id,),
            ).fetchall()
        return {str(r["k"] or none_label): int(r["n"]) for r in rows}

    with db.connect() as conn:
        problem_count = int(
            conn.execute(
                f"SELECT COUNT(*) AS n FROM problems WHERE {run_scope('problems')}",
                (run_id,),
            ).fetchone()["n"]
        )
        ir_count = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM incoming_requests WHERE run_id = ?",
                (run_id,),
            ).fetchone()["n"]
        )
        res_count = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM resolutions WHERE run_id = ?",
                (run_id,),
            ).fetchone()["n"]
        )
        trace_count = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM agent_traces WHERE run_id = ?",
                (run_id,),
            ).fetchone()["n"]
        )

    return {
        "problem_count": problem_count,
        "incoming_requests_count": ir_count,
        "resolutions_count": res_count,
        "agent_traces_count": trace_count,
        "complexity_counts": _group_count("problems", "complexity"),
        # Per case (lineage row), so multi-contact cases are not counted twice.
        "type_counts": _group_count("lineage", "ticket_type"),
        "tier_counts": _group_count("lineage", "customer_tier"),
        "tone_counts": _group_count("lineage", "customer_tone"),
        "channel_counts": _group_count("resolutions", "channel"),
        "contacts_per_case": _group_count("resolutions", "round_count", "AND round_index = 1"),
        # Honest outcomes: the state each case ended in (its last contact) and how contacts ended.
        "problem_state_counts": _group_count(
            "resolutions", "problem_state", "AND round_index = round_count", "unassessed"
        ),
        "contact_ending_counts": _group_count(
            "resolutions", "contact_ending", none_label="unassessed"
        ),
        "quality_flags": {
            "problems": _group_count("problems", "quality_flag"),
            "incoming_requests": _group_count("incoming_requests", "quality_flag"),
            "resolutions": _group_count("resolutions", "quality_flag"),
        },
        "duration_s": round((datetime.now(UTC) - started_at).total_seconds(), 3),
    }


async def _acompute_run_stats(
    *,
    adb: AsyncDatabase,
    run_id: str,
    started_at: datetime,
) -> dict[str, Any]:
    """Async sibling of :func:`_compute_run_stats` used inside graph nodes."""

    async def _group_count(
        table: str, group_col: str, extra_where: str = "", none_label: str = "ok"
    ) -> dict[str, int]:
        async with adb.connect() as conn:
            cur = await conn.execute(
                f"SELECT {group_col} AS k, COUNT(*) AS n FROM {table} "
                f"WHERE {run_scope(table)} {extra_where} GROUP BY {group_col} ORDER BY {group_col}",
                (run_id,),
            )
            rows = await cur.fetchall()
        return {str(r["k"] or none_label): int(r["n"]) for r in rows}

    async def _count(table: str) -> int:
        async with adb.connect() as conn:
            cur = await conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE {run_scope(table)}",
                (run_id,),
            )
            row = await cur.fetchone()
        assert row is not None
        return int(row["n"])

    return {
        "problem_count": await _count("problems"),
        "incoming_requests_count": await _count("incoming_requests"),
        "resolutions_count": await _count("resolutions"),
        "agent_traces_count": await _count("agent_traces"),
        "complexity_counts": await _group_count("problems", "complexity"),
        # Per case (lineage row), so multi-contact cases are not counted twice.
        "type_counts": await _group_count("lineage", "ticket_type"),
        "tier_counts": await _group_count("lineage", "customer_tier"),
        "tone_counts": await _group_count("lineage", "customer_tone"),
        "channel_counts": await _group_count("resolutions", "channel"),
        "contacts_per_case": await _group_count(
            "resolutions", "round_count", "AND round_index = 1"
        ),
        "problem_state_counts": await _group_count(
            "resolutions", "problem_state", "AND round_index = round_count", "unassessed"
        ),
        "contact_ending_counts": await _group_count(
            "resolutions", "contact_ending", none_label="unassessed"
        ),
        "quality_flags": {
            "problems": await _group_count("problems", "quality_flag"),
            "incoming_requests": await _group_count("incoming_requests", "quality_flag"),
            "resolutions": await _group_count("resolutions", "quality_flag"),
        },
        "duration_s": round((datetime.now(UTC) - started_at).total_seconds(), 3),
    }


# --------------------------------------------------------------------------- #
# Pipeline entry-point
# --------------------------------------------------------------------------- #


async def run_pipeline(
    *,
    settings: AppSettings,
    factory: AgentFactory,
    db: Database,
    company: CompanyProfile,
    scenarios: ScenarioCatalogue,
    case_plan: CasePlan | None = None,
    problems_from: str | None = None,
) -> str:
    """Run the full deterministic pipeline; returns the new run_id.

    ``case_plan`` pins dimensions per case (see csfd.case_plan).
    ``problems_from`` names an earlier run whose committed problems (in the same
    database) are reused: Phase 1 is skipped and the run is recorded as
    ``phase='phase2'`` with that ``parent_run_id``.

    Builds the LangGraph parent graph (composed of Phase 1 + Phase 2
    subgraphs), opens an async SQLite checkpointer for durable execution, and
    ``ainvoke``s the graph once with ``thread_id=run_id``. Persistence and
    agent-trace recording all happen inside graph node functions.
    """
    from csfd.graph.checkpointer import async_sqlite_checkpointer
    from csfd.graph.pipeline_graph import PipelineState, build_pipeline_graph

    run_id = str(uuid4())
    Path(settings.storage.checkpoint_sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    _log.info("pipeline.start", run_id=run_id)
    async with async_sqlite_checkpointer(settings.storage.checkpoint_sqlite_path) as checkpointer:
        graph = build_pipeline_graph(
            factory=factory,
            db=db,
            settings=settings,
            checkpointer=checkpointer,
        )
        state = PipelineState(
            run_id=run_id,
            run_seed=settings.pipeline.run_seed or 0,
            company=company,
            scenarios=scenarios,
            validation_enabled=settings.validation.enabled,
            max_retries=settings.validation.max_retries,
            case_plan=case_plan,
            parent_run_id=problems_from,
            problems_committed=(
                ProblemRepo(db).list_for_run(problems_from) if problems_from else []
            ),
        )
        await graph.ainvoke(state, config={"configurable": {"thread_id": run_id}})
    _log.info("pipeline.end", run_id=run_id)
    return run_id
