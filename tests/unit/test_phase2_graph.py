"""Unit tests for the Phase 2 subgraph routers and builder shape.

These tests pin the conditional-edge dispatch logic and confirm that
``build_phase2_subgraph`` wires the expected node names. The full pipeline
end-to-end behaviour is covered by
``tests/integration/test_pipeline_deterministic.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.graph.phase2_graph import (
    _route_after_commit_resolution,
    _route_after_generate_resolution,
    _route_after_validate_resolution,
    build_phase2_subgraph,
)
from csfd.graph.pipeline_graph import PipelineState, _PlanSlot
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
from csfd.ticket_types.definitions import TicketType


def _company() -> CompanyProfile:
    return CompanyProfile(name="X", raw_markdown="x", sections={})


def _scenarios() -> ScenarioCatalogue:
    return ScenarioCatalogue(scenarios=[], source_path=Path("seeds/scenarios_seed.md"))


def _make_state(**overrides: Any) -> PipelineState:
    """Build a ``PipelineState`` with minimal required fields and optional overrides."""
    defaults: dict[str, Any] = {
        "run_id": "test-run",
        "run_seed": 0,
        "company": _company(),
        "scenarios": _scenarios(),
        "validation_enabled": True,
        "max_retries": 2,
    }
    defaults.update(overrides)
    return PipelineState(**defaults)


def _plan_slot(index: int) -> _PlanSlot:
    return _PlanSlot(
        index=index,
        problem_id=f"p{index:03d}",
        ticket_type=TicketType.DOCS_REQUEST,
        tier="standard",
        tone="neutral",
    )


def _build_stub_settings() -> AppSettings:
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
        storage=StorageConfig(sqlite_path=":memory:"),
    )


def _build_stub_factory() -> AgentFactory:
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


# --------------------------------------------------------------------------- #
# Router tests
# --------------------------------------------------------------------------- #


def test_route_after_generate_when_validation_enabled() -> None:
    state = _make_state(validation_enabled=True)
    assert _route_after_generate_resolution(state) == "validate"


def test_route_after_generate_when_validation_disabled() -> None:
    state = _make_state(validation_enabled=False)
    assert _route_after_generate_resolution(state) == "commit"


def test_route_after_validate_pass() -> None:
    state = _make_state(
        last_verdict=Verdict(checker="x", passed=True, issues=[]),
        retry_attempt=0,
        max_retries=2,
    )
    assert _route_after_validate_resolution(state) == "pass"


def test_route_after_validate_retry() -> None:
    state = _make_state(
        last_verdict=Verdict(checker="x", passed=False, issues=[]),
        retry_attempt=1,
        max_retries=3,
    )
    assert _route_after_validate_resolution(state) == "retry"


def test_route_after_validate_exhausted() -> None:
    state = _make_state(
        last_verdict=Verdict(checker="x", passed=False, issues=[]),
        retry_attempt=2,
        max_retries=2,
    )
    assert _route_after_validate_resolution(state) == "exhausted"


def test_route_after_commit_with_more_slots() -> None:
    state = _make_state(
        plan_slots=[_plan_slot(0), _plan_slot(1), _plan_slot(2)],
        slot_index=1,
    )
    assert _route_after_commit_resolution(state) == "next"


def test_route_after_commit_done() -> None:
    state = _make_state(
        plan_slots=[_plan_slot(0), _plan_slot(1)],
        slot_index=2,
    )
    assert _route_after_commit_resolution(state) == "done"


# --------------------------------------------------------------------------- #
# Builder shape
# --------------------------------------------------------------------------- #


def test_build_phase2_subgraph_compiles_with_expected_nodes(tmp_path: Path) -> None:
    settings = _build_stub_settings()
    factory = _build_stub_factory()
    db = Database(path=tmp_path / "x.sqlite")

    compiled = build_phase2_subgraph(factory=factory, db=db, settings=settings)

    assert isinstance(compiled, CompiledStateGraph)
    node_names = set(compiled.get_graph().nodes.keys())
    expected = {
        "build_allocation_plan",
        "generate_resolution",
        "validate_resolution",
        "commit_resolution",
        "_bump_retry",
        "_mark_exhausted",
        "_mark_validation_skipped",
    }
    assert expected.issubset(node_names)
