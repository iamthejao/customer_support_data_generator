"""LangGraph implementation of the deterministic CSFD pipeline.

Composes two phase subgraphs into a parent graph:

    START → init_run → [Phase 1 subgraph] → [Phase 2 subgraph] → finalize_run → END

* `init_run`        — create the `runs` row, capture `git_sha` + `started_at`,
                      seed run-scoped state fields.
* Phase 1 subgraph  — Problem Database generation with the per-problem
                      generator → checker → retry loop. Defined in
                      ``csfd.graph.phase1_graph``.
* Phase 2 subgraph  — Allocation plan + per-slot resolution generation with
                      the same retry shape. Defined in ``csfd.graph.phase2_graph``.
* `finalize_run`    — compute run stats and mark the run completed.

All three graphs share a single :class:`PipelineState` Pydantic model so the
parent's state flows straight into both subgraphs without input/output
mapping.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.models.registry import build_llm
from csfd.pipeline import (
    ProblemBrainstormOutput,
    ResolutionOutput,
    _acompute_run_stats,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.settings import AppSettings, load_settings
from csfd.storage.db import Database
from csfd.storage.db_async import AsyncDatabase
from csfd.storage.repository import ProblemEmbeddingRecord, ProblemRecord, RunRecord, RunRepo
from csfd.ticket_types.definitions import TicketType
from csfd.utils.git import current_git_sha

type _CompiledGraph = CompiledStateGraph[Any, Any, Any, Any]


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


class _PlanSlot(BaseModel):
    """Pydantic mirror of ``csfd.allocator.TicketSlot`` for checkpoint serialization."""

    index: int
    problem_id: str
    ticket_type: TicketType
    tier: str
    tone: str


class PipelineState(BaseModel):
    """Shared state for the parent graph and both phase subgraphs.

    Field ownership is documented inline. Each layer reads/writes only its own
    fields; using a single model avoids LangGraph input/output mapping.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- constants for the run (set by caller + init_run) ---
    run_id: str
    run_seed: int
    started_at: datetime | None = None  # populated by init_run
    company: CompanyProfile
    scenarios: ScenarioCatalogue
    validation_enabled: bool
    max_retries: int

    # --- Phase 1 progress ---
    target_complexities: list[str] = Field(default_factory=list)
    problem_index: int = 0
    problems_committed: list[ProblemRecord] = Field(default_factory=list)
    problem_embeddings_committed: list[ProblemEmbeddingRecord] = Field(default_factory=list)
    dedup_enabled: bool = False

    # --- Phase 2 progress ---
    plan_slots: list[_PlanSlot] = Field(default_factory=list)
    slot_index: int = 0

    # --- per-artifact transient (reused across both phases; reset in commit_*) ---
    retry_attempt: int = 0
    current_problem_draft: ProblemBrainstormOutput | None = None
    current_resolution_draft: ResolutionOutput | None = None
    current_problem_embedding: list[float] | None = None
    # Full verdict (not just the bool) so the next retry can feed `issues` into
    # the prompt's "prior attempt failed validation" block.
    last_verdict: Verdict | None = None
    last_quality_flag: str | None = None
    last_generator_trace_id: str | None = None


# --------------------------------------------------------------------------- #
# Parent nodes
# --------------------------------------------------------------------------- #


async def init_run_node(
    state: PipelineState,
    *,
    db: Database,
    adb: AsyncDatabase,
    settings: AppSettings,
) -> dict[str, Any]:
    """Create the ``runs`` row at start of pipeline. Stamps ``git_sha``."""
    import json

    started_at = datetime.now(UTC)
    await RunRepo(db).acreate(
        adb,
        RunRecord(
            id=state.run_id,
            phase="full",
            parent_run_id=None,
            status="running",
            started_at=started_at,
            completed_at=None,
            run_seed=state.run_seed,
            pipeline_version=settings.pipeline.version,
            git_sha=current_git_sha(),
            config_snapshot_json=json.dumps(
                {
                    "problem_database": settings.problem_database.model_dump(),
                    "tickets": settings.tickets.model_dump(),
                    "validation": settings.validation.model_dump(),
                    "embedding": settings.embedding.model_dump(),
                },
                ensure_ascii=False,
            ),
            stats_json=None,
            error_summary=None,
        ),
    )
    return {"started_at": started_at}


async def finalize_run_node(
    state: PipelineState,
    *,
    db: Database,
    adb: AsyncDatabase,
) -> dict[str, Any]:
    """Compute end-of-run stats and mark the run completed."""
    assert state.started_at is not None, "init_run must set started_at"
    stats = await _acompute_run_stats(adb=adb, run_id=state.run_id, started_at=state.started_at)
    await RunRepo(db).aupdate_status(
        adb, state.run_id, status="completed", completed=True, stats=stats
    )
    return {}


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def build_pipeline_graph(
    *,
    factory: AgentFactory,
    db: Database,
    settings: AppSettings,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    adb: AsyncDatabase | None = None,
) -> _CompiledGraph:
    """Build and compile the parent pipeline graph.

    The two phase subgraphs are imported lazily so this module remains
    importable while ``phase1_graph`` / ``phase2_graph`` are being authored.
    An ``AsyncDatabase`` is constructed from ``db.path`` if not supplied; it
    is used by every graph node body for sqlite writes (the sync ``db`` is
    still used for read paths that don't need to be async-safe).
    """
    from csfd.embeddings.client import OpenAICompatEmbedder
    from csfd.graph.phase1_graph import build_phase1_subgraph
    from csfd.graph.phase2_graph import build_phase2_subgraph

    adb_local = adb if adb is not None else AsyncDatabase(db.path)
    embedder = None
    if settings.embedding.enabled:
        embedder = OpenAICompatEmbedder(
            base_url=settings.embedding.base_url,
            api_key="ollama" if "localhost" in settings.embedding.base_url else None,
            model=settings.embedding.model,
            dim=settings.embedding.dim,
            timeout_s=settings.embedding.timeout_s,
        )
    g: StateGraph[PipelineState, Any, PipelineState, PipelineState] = StateGraph(PipelineState)
    g.add_node("init_run", partial(init_run_node, db=db, adb=adb_local, settings=settings))
    g.add_node(
        "phase1",
        build_phase1_subgraph(
            factory=factory, db=db, adb=adb_local, settings=settings, embedder=embedder
        ),
    )
    g.add_node(
        "phase2", build_phase2_subgraph(factory=factory, db=db, adb=adb_local, settings=settings)
    )
    g.add_node("finalize_run", partial(finalize_run_node, db=db, adb=adb_local))

    g.add_edge(START, "init_run")
    g.add_edge("init_run", "phase1")
    g.add_edge("phase1", "phase2")
    g.add_edge("phase2", "finalize_run")
    g.add_edge("finalize_run", END)

    return g.compile(checkpointer=checkpointer)


def _factory_from_settings() -> AgentFactory:
    """Build an ``AgentFactory`` from the on-disk default settings + prompts.

    Used exclusively by the LangGraph Studio entrypoint below — production
    callers wire the factory + db themselves.
    """
    settings = load_settings()
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=build_llm,
        agent_configs=settings.agents,
    )


def make_pipeline_studio_graph() -> _CompiledGraph:
    """LangGraph Studio entrypoint — pre-built at module import time.

    No checkpointer is wired here: the LangGraph API/Studio runtime provides
    persistence on its own and rejects graphs that carry a custom
    checkpointer. CLI runs (see ``csfd.pipeline.run_pipeline``) use the
    ``async_sqlite_checkpointer`` for durable execution.
    """
    settings = load_settings()
    factory = _factory_from_settings()
    db = Database(path=Path(settings.storage.sqlite_path))
    return build_pipeline_graph(
        factory=factory,
        db=db,
        settings=settings,
        checkpointer=None,
    )
