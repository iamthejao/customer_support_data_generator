"""Unit tests for the parent pipeline graph wiring and parent-node side effects.

These tests pin the ``PipelineState`` defaults, exercise ``init_run_node`` and
``finalize_run_node`` directly against a real SQLite database, and confirm
that the parent builder + Studio entrypoint return compiled graphs with the
expected node names. End-to-end pipeline behaviour is covered by
``tests/integration/test_pipeline_deterministic.py``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jinja2
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from csfd.agents.factory import AgentFactory
from csfd.graph.pipeline_graph import (
    PipelineState,
    build_pipeline_graph,
    finalize_run_node,
    init_run_node,
    make_pipeline_studio_graph,
)
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle, PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.settings import (
    AgentLLMConfig,
    AppSettings,
    BudgetConfig,
    ObservabilityConfig,
    PipelineConfig,
    ProblemDatabaseConfig,
    StorageConfig,
    TicketsConfig,
    ValidationConfig,
)
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import RunRecord, RunRepo


def _company() -> CompanyProfile:
    return CompanyProfile(name="X", raw_markdown="x", sections={})


def _scenarios() -> ScenarioCatalogue:
    return ScenarioCatalogue(scenarios=[], source_path=Path("seeds/scenarios_seed.md"))


def _build_settings(*, sqlite_path: Path) -> AppSettings:
    return AppSettings(
        pipeline=PipelineConfig(version="test", run_seed=0, budget=BudgetConfig()),
        agents={"generator": AgentLLMConfig(provider="anthropic", model="stub")},
        problem_database=ProblemDatabaseConfig(count=1, complexity_proportions={"simple": 1.0}),
        tickets=TicketsConfig(
            total=1,
            type_proportions={"docs_request": 1.0, "l1": 0.0, "l2": 0.0, "l3": 0.0},
            assignment_strategy="complexity_weighted",
            turns_per_type={"docs_request": 2, "l1": 3, "l2": 5, "l3": 7},
            tier_proportions={"standard": 1.0},
            tone_proportions_per_type={
                "docs_request": {"neutral": 1.0},
                "l1": {"neutral": 1.0},
                "l2": {"neutral": 1.0},
                "l3": {"neutral": 1.0},
            },
        ),
        validation=ValidationConfig(enabled=False, max_retries=0),
        observability=ObservabilityConfig(),
        storage=StorageConfig(sqlite_path=str(sqlite_path)),
    )


def _build_factory() -> AgentFactory:
    env = jinja2.Environment(autoescape=False)
    handle = PromptHandle(
        name="stub",
        template=env.from_string("stub"),
        source="stub",
        version="stub",
    )
    reg = PromptRegistry(root=Path("/dev/null"))
    reg._handles = {
        "phase1.problem_brainstorm_v2": handle,
        "phase1.problem_combined_check": handle,
        "phase2.resolution_generator": handle,
        "phase2.resolution_combined_check": handle,
    }
    fake = FakeChatModel(structured={})

    def _builder(_cfg: AgentLLMConfig) -> BaseChatModel:
        return fake

    return AgentFactory(
        prompts=reg,
        llm_builder=_builder,
        agent_configs={
            "generator": AgentLLMConfig(provider="anthropic", model="stub"),
            "combined_checker": AgentLLMConfig(provider="anthropic", model="stub"),
        },
    )


def _make_state(**overrides: Any) -> PipelineState:
    defaults: dict[str, Any] = {
        "run_id": "test-run",
        "run_seed": 0,
        "company": _company(),
        "scenarios": _scenarios(),
        "validation_enabled": False,
        "max_retries": 0,
    }
    defaults.update(overrides)
    return PipelineState(**defaults)


# --------------------------------------------------------------------------- #
# State defaults
# --------------------------------------------------------------------------- #


def test_pipeline_state_required_fields() -> None:
    state = PipelineState(
        run_id="r1",
        run_seed=42,
        company=_company(),
        scenarios=_scenarios(),
        validation_enabled=True,
        max_retries=3,
    )
    assert state.run_id == "r1"
    assert state.run_seed == 42
    assert state.validation_enabled is True
    assert state.max_retries == 3
    # Defaults for progress / transient fields.
    assert state.started_at is None
    assert state.retry_attempt == 0
    assert state.problem_index == 0
    assert state.slot_index == 0
    assert state.target_complexities == []
    assert state.plan_slots == []
    assert state.problems_committed == []
    assert state.current_problem_draft is None
    assert state.current_resolution_draft is None
    assert state.last_verdict is None
    assert state.last_quality_flag is None
    assert state.last_generator_trace_id is None


# --------------------------------------------------------------------------- #
# Parent-node side effects
# --------------------------------------------------------------------------- #


def test_init_run_node_writes_run_record(tmp_path: Path) -> None:
    settings = _build_settings(sqlite_path=tmp_path / "runs.sqlite")
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)

    state = _make_state(run_id="r1")
    result = asyncio.run(init_run_node(state, db=db, settings=settings))

    # The node returns the started_at it just wrote so LangGraph can merge it.
    assert "started_at" in result
    assert isinstance(result["started_at"], datetime)

    run = RunRepo(db).get("r1")
    assert run.status == "running"
    # git_sha is captured from the current working tree; tests run from inside
    # the repo so it should be a 40-char SHA, but allow None for sandboxes
    # where the subprocess fails.
    if run.git_sha is not None:
        assert len(run.git_sha) == 40
    assert run.config_snapshot_json  # non-empty
    snapshot = json.loads(run.config_snapshot_json)
    assert set(snapshot.keys()) == {"problem_database", "tickets", "validation"}
    assert run.stats_json is None  # not populated until finalize


def test_finalize_run_node_writes_stats(tmp_path: Path) -> None:
    settings = _build_settings(sqlite_path=tmp_path / "runs.sqlite")
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)

    # Seed a "running" run row directly — bypass init_run_node so we control
    # started_at and assert a non-trivial duration.
    started_at = datetime.now(UTC) - timedelta(seconds=5)
    RunRepo(db).create(
        RunRecord(
            id="r1",
            phase="full",
            parent_run_id=None,
            status="running",
            started_at=started_at,
            completed_at=None,
            run_seed=0,
            pipeline_version="test",
            git_sha=None,
            config_snapshot_json="{}",
            stats_json=None,
            error_summary=None,
        )
    )

    state = _make_state(run_id="r1", started_at=started_at)
    result = asyncio.run(finalize_run_node(state, db=db))
    assert result == {}

    run = RunRepo(db).get("r1")
    assert run.status == "completed"
    assert run.stats_json is not None
    stats = json.loads(run.stats_json)
    assert isinstance(stats, dict)
    expected_keys = {
        "problem_count",
        "incoming_requests_count",
        "resolutions_count",
        "agent_traces_count",
        "complexity_counts",
        "type_counts",
        "tier_counts",
        "tone_counts",
        "quality_flags",
        "duration_s",
    }
    assert expected_keys.issubset(stats.keys())
    # All counts are 0 because we never ran the phase subgraphs.
    assert stats["problem_count"] == 0
    assert stats["incoming_requests_count"] == 0
    assert stats["resolutions_count"] == 0
    # We seeded started_at 5s in the past so duration_s must be positive.
    assert stats["duration_s"] > 0


# --------------------------------------------------------------------------- #
# Builder shape
# --------------------------------------------------------------------------- #


def test_build_pipeline_graph_compiles_with_expected_nodes(tmp_path: Path) -> None:
    settings = _build_settings(sqlite_path=tmp_path / "runs.sqlite")
    db = Database(path=Path(settings.storage.sqlite_path))
    factory = _build_factory()

    compiled = build_pipeline_graph(factory=factory, db=db, settings=settings, checkpointer=None)
    assert isinstance(compiled, CompiledStateGraph)
    node_names = set(compiled.get_graph().nodes.keys())
    assert {"init_run", "phase1", "phase2", "finalize_run"}.issubset(node_names)


def test_make_pipeline_studio_graph_returns_compiled_graph() -> None:
    # Relies on config/default.yaml + prompts/ being present at CWD, which the
    # test runner provides.
    compiled = make_pipeline_studio_graph()
    assert isinstance(compiled, CompiledStateGraph)
    node_names = set(compiled.get_graph().nodes.keys())
    assert {"init_run", "phase1", "phase2", "finalize_run"}.issubset(node_names)
