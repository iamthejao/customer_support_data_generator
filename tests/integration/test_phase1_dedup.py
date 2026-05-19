"""Phase 1 dedup integration tests.

Validates the two dedup contracts the README promises:

* Test 1: A near-duplicate candidate triggers the retry loop and the matched
  problem's title + summary are surfaced to the generator on the next attempt
  via ``AgentContext.prior_verdicts``.
* Test 2: When retries are exhausted while still hitting duplicates, the
  near-duplicate is committed with ``quality_flag = "warning:dedup_exhausted"``
  (no stall).

These tests run the Phase 1 subgraph directly (no Phase 2) for speed/isolation
and wire a :class:`FakeEmbedder` via the ``embedder=`` kwarg.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from csfd.agents import tracing as tracing_mod
from csfd.agents.base import AgentContext, Verdict
from csfd.agents.factory import AgentFactory
from csfd.agents.tracing import TracingAdapter
from csfd.embeddings.fake import FakeEmbedder
from csfd.graph.phase1_graph import build_phase1_subgraph
from csfd.graph.pipeline_graph import PipelineState
from csfd.models.fake import FakeChatModel
from csfd.pipeline import ProblemBrainstormOutput
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import Scenario, ScenarioCatalogue
from csfd.settings import (
    AgentLLMConfig,
    AppSettings,
    BudgetConfig,
    EmbeddingConfig,
    ObservabilityConfig,
    PipelineConfig,
    ProblemDatabaseConfig,
    StorageConfig,
    TicketsConfig,
    ValidationConfig,
)
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemEmbeddingRepo,
    RunRecord,
    RunRepo,
)
from csfd.ticket_types.definitions import ProblemComplexity

# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class _SequentialProblemFake(FakeChatModel):
    """Returns a configurable sequence of ProblemBrainstormOutput / always-pass Verdict.

    The generator's structured-output runnable pops one entry from
    ``problem_outputs`` per call. The checker always returns ``passed=True``
    so the only failure path exercised is the dedup node.
    """

    problem_outputs: list[ProblemBrainstormOutput] = Field(default_factory=list)
    problem_call_idx: int = 0

    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel], **kw: Any
    ) -> Runnable[Any, BaseModel]:
        if schema is Verdict:
            return RunnableLambda(lambda _i: Verdict(checker="fake_pass", passed=True, issues=[]))
        if schema is ProblemBrainstormOutput:

            def _next(_inputs: Any) -> BaseModel:
                idx = self.problem_call_idx
                self.problem_call_idx += 1
                # Clamp to last entry so over-calls don't IndexError -- the
                # assertion on call count is what enforces the contract.
                out = self.problem_outputs[min(idx, len(self.problem_outputs) - 1)]
                return out

            return RunnableLambda(_next)
        raise KeyError(schema)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _settings(
    tmp_path: Path,
    *,
    max_retries: int,
    embedding_enabled: bool = True,
    threshold: float = 0.99,
    dim: int = 4,
) -> AppSettings:
    return AppSettings(
        pipeline=PipelineConfig(version="t", run_seed=1, budget=BudgetConfig()),
        agents={},
        problem_database=ProblemDatabaseConfig(
            count=2,
            complexity_proportions={"simple": 1.0, "medium": 0.0, "complex": 0.0},
        ),
        tickets=TicketsConfig(
            total=0,
            type_proportions={"docs_request": 1.0, "l1": 0.0, "l2": 0.0, "l3": 0.0},
            assignment_strategy="complexity_weighted",
            turns_per_type={"docs_request": 2, "l1": 3, "l2": 5, "l3": 7},
            tier_proportions={"standard": 1.0, "premium": 0.0, "enterprise": 0.0},
            tone_proportions_per_type={
                "docs_request": {"neutral": 1.0},
                "l1": {"neutral": 1.0},
                "l2": {"neutral": 1.0},
                "l3": {"neutral": 1.0},
            },
        ),
        validation=ValidationConfig(enabled=True, max_retries=max_retries),
        embedding=EmbeddingConfig(enabled=embedding_enabled, dim=dim, threshold=threshold),
        observability=ObservabilityConfig(),
        storage=StorageConfig(sqlite_path=str(tmp_path / "runs.sqlite")),
    )


def _factory(fake: BaseChatModel) -> AgentFactory:
    def _builder(_cfg: AgentLLMConfig) -> BaseChatModel:
        return fake

    dummy = AgentLLMConfig(provider="anthropic", model="fake")
    configs = {
        n: dummy
        for n in (
            "generator",
            "combined_checker",
            "problem_brainstorm",
            "combined_problem_check",
            "resolution_generator",
            "combined_resolution_check",
        )
    }
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(prompts=reg, llm_builder=_builder, agent_configs=configs)


@pytest.fixture
def company() -> CompanyProfile:
    return CompanyProfile(name="Acme", raw_markdown="# Acme", sections={})


@pytest.fixture
def scenarios() -> ScenarioCatalogue:
    return ScenarioCatalogue(
        scenarios=[Scenario(category="auth", title="x", summary="x")],
        source_path=Path("seeds/scenarios_seed.md"),
    )


def _seed_run(db: Database, run_id: str) -> None:
    """Seed a ``runs`` row so the foreign-keyed ``problems.run_id`` insert succeeds."""
    RunRepo(db).create(
        RunRecord(
            id=run_id,
            phase="phase1",
            parent_run_id=None,
            status="running",
            started_at=datetime.now(UTC),
            completed_at=None,
            run_seed=1,
            pipeline_version="t",
            git_sha=None,
            config_snapshot_json=json.dumps({}),
            stats_json=None,
            error_summary=None,
        )
    )


def _initial_state(
    *,
    run_id: str,
    company: CompanyProfile,
    scenarios: ScenarioCatalogue,
    settings: AppSettings,
) -> PipelineState:
    return PipelineState(
        run_id=run_id,
        run_seed=settings.pipeline.run_seed,
        company=company,
        scenarios=scenarios,
        validation_enabled=settings.validation.enabled,
        max_retries=settings.validation.max_retries,
    )


def _problem(title: str, summary: str) -> ProblemBrainstormOutput:
    return ProblemBrainstormOutput(
        title=title,
        summary=summary,
        background="bg",
        category="auth",
        complexity=ProblemComplexity.SIMPLE,
        resolution_hints={"docs_request": "x", "l1": "x", "l2": "x", "l3": "x"},
    )


# --------------------------------------------------------------------------- #
# Test 1: duplicate triggers retry; matched problem fed back to the generator
# --------------------------------------------------------------------------- #


def test_duplicate_triggers_retry_and_feeds_matched_problem_to_generator(
    tmp_path: Path,
    company: CompanyProfile,
    scenarios: ScenarioCatalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path, max_retries=1)

    fake = _SequentialProblemFake()
    fake.problem_outputs = [
        _problem("Problem A", "Problem A summary"),  # slot 0, attempt 0
        _problem("Problem A", "Problem A summary"),  # slot 1, attempt 0 -> duplicate
        _problem("Problem B", "Problem B summary"),  # slot 1, attempt 1 -> unique
    ]

    embedder = FakeEmbedder(
        dim=4,
        force={
            "Problem A\n\nProblem A summary": (1.0, 0.0, 0.0, 0.0),
            "Problem B\n\nProblem B summary": (0.0, 1.0, 0.0, 0.0),
        },
    )

    # Capture every AgentContext passed to a generator-role TracingAdapter.
    captured_generator_ctx: list[AgentContext] = []
    original_invoke = TracingAdapter.invoke

    async def _spy_invoke(self: TracingAdapter, ctx: AgentContext) -> BaseModel:
        if getattr(self, "_role", None) == "generator":
            captured_generator_ctx.append(ctx)
        return await original_invoke(self, ctx)

    monkeypatch.setattr(tracing_mod.TracingAdapter, "invoke", _spy_invoke)

    factory = _factory(fake)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    run_id = "run-dedup-1"
    _seed_run(db, run_id)

    compiled = build_phase1_subgraph(factory=factory, db=db, settings=settings, embedder=embedder)
    state = _initial_state(run_id=run_id, company=company, scenarios=scenarios, settings=settings)

    asyncio.run(compiled.ainvoke(state))

    # The generator should have been called 3 times: slot0 once, slot1 twice.
    assert (
        len(captured_generator_ctx) == 3
    ), f"expected 3 generator calls, got {len(captured_generator_ctx)}"

    # The third call (slot 1, retry attempt 1) is the key contract: it must
    # have received the dedup verdict as a prior_verdict so the prompt's
    # "prior attempt failed" block can surface the matched title + summary.
    retry_ctx = captured_generator_ctx[2]
    assert retry_ctx.retry_attempt == 1
    assert len(retry_ctx.prior_verdicts) == 1
    issue = retry_ctx.prior_verdicts[0].issues[0]
    assert issue.rule_violated == "near_duplicate_of_committed_problem"
    assert "Problem A" in issue.explanation
    assert "Problem A summary" in issue.explanation

    # Both committed problems have embeddings persisted with the configured dim.
    embeddings = ProblemEmbeddingRepo(db).list_for_run(run_id)
    assert len(embeddings) == 2
    assert all(e.dim == 4 for e in embeddings)


# --------------------------------------------------------------------------- #
# Test 2: exhausted retries commits with warning:dedup_exhausted
# --------------------------------------------------------------------------- #


def test_exhausted_dedup_retries_commits_with_warning_flag(
    tmp_path: Path,
    company: CompanyProfile,
    scenarios: ScenarioCatalogue,
) -> None:
    settings = _settings(tmp_path, max_retries=0)

    fake = _SequentialProblemFake()
    # Always emit the same problem text -- every candidate after slot 0 will
    # collide with the committed slot-0 embedding.
    fake.problem_outputs = [_problem("Same", "Same summary")]

    embedder = FakeEmbedder(
        dim=4,
        force={"Same\n\nSame summary": (1.0, 0.0, 0.0, 0.0)},
    )

    factory = _factory(fake)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    run_id = "run-dedup-2"
    _seed_run(db, run_id)

    compiled = build_phase1_subgraph(factory=factory, db=db, settings=settings, embedder=embedder)
    state = _initial_state(run_id=run_id, company=company, scenarios=scenarios, settings=settings)

    asyncio.run(compiled.ainvoke(state))

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT quality_flag FROM problems WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()

    # Both problems committed -- no stall.
    assert len(rows) == 2
    # Slot 0 has no flag (clean commit); slot 1 was a duplicate with no
    # retry budget, so it commits with warning:dedup_exhausted.
    assert rows[0]["quality_flag"] is None
    assert rows[1]["quality_flag"] == "warning:dedup_exhausted"
