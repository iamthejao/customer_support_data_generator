"""Unit tests for the Phase 2 subgraph routers and builder shape.

These tests pin the conditional-edge dispatch logic and confirm that
``build_phase2_subgraph`` wires the expected node names. The full pipeline
end-to-end behaviour is covered by
``tests/integration/test_pipeline_deterministic.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import jinja2
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.graph.phase2_graph import (
    _mark_cap_hit_node,
    _route_after_agent_turn,
    _route_after_build_allocation_plan,
    _route_after_commit_resolution,
    _route_after_customer_turn,
    _route_after_validate_conversation,
    build_phase2_subgraph,
)
from csfd.graph.pipeline_graph import PipelineState, _PlanSlot, _RoundSpec
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle, PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.settings import (
    AgentLLMConfig,
    AppSettings,
    BudgetConfig,
    DialogueConfig,
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
            dialogue=DialogueConfig(turn_cap=20),
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
        "phase2.incoming_request": handle,
        "phase2.customer_turn": handle,
        "phase2.agent_turn": handle,
        "phase2.conversation_consistency_check": handle,
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


def _dialogue_turns(n: int) -> list[Any]:
    from csfd.pipeline import DialogueTurnOutput

    return [
        DialogueTurnOutput(speaker="customer" if i % 2 == 0 else "agent", content=f"t{i}")
        for i in range(n)
    ]


def test_route_after_agent_turn_done() -> None:
    state = _make_state(dialogue_done=True, current_dialogue_turns=_dialogue_turns(4))
    assert _route_after_agent_turn(state) == "consistency"


def test_route_after_agent_turn_continue() -> None:
    state = _make_state(dialogue_done=False, current_dialogue_turns=_dialogue_turns(4))
    assert _route_after_agent_turn(state) == "customer"


def test_route_after_agent_turn_cap() -> None:
    state = _make_state(
        dialogue_done=False, current_dialogue_turns=_dialogue_turns(20), dialogue_turn_cap=20
    )
    assert _route_after_agent_turn(state) == "cap"


def test_route_after_customer_turn_done() -> None:
    state = _make_state(dialogue_done=True, current_dialogue_turns=_dialogue_turns(3))
    assert _route_after_customer_turn(state) == "consistency"


def test_route_after_customer_turn_continue() -> None:
    state = _make_state(dialogue_done=False, current_dialogue_turns=_dialogue_turns(3))
    assert _route_after_customer_turn(state) == "agent"


def test_route_after_customer_turn_cap() -> None:
    state = _make_state(
        dialogue_done=False, current_dialogue_turns=_dialogue_turns(20), dialogue_turn_cap=20
    )
    assert _route_after_customer_turn(state) == "cap"


def test_route_after_validate_pass() -> None:
    state = _make_state(
        last_verdict=Verdict(checker="x", passed=True, issues=[]),
        retry_attempt=0,
        max_retries=2,
    )
    assert _route_after_validate_conversation(state) == "pass"


def test_route_after_validate_retry() -> None:
    state = _make_state(
        last_verdict=Verdict(checker="x", passed=False, issues=[]),
        retry_attempt=1,
        max_retries=3,
    )
    assert _route_after_validate_conversation(state) == "retry"


def test_route_after_validate_exhausted() -> None:
    state = _make_state(
        last_verdict=Verdict(checker="x", passed=False, issues=[]),
        retry_attempt=2,
        max_retries=2,
    )
    assert _route_after_validate_conversation(state) == "exhausted"


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


def test_route_after_build_allocation_plan_done_when_empty() -> None:
    state = _make_state(plan_slots=[])
    assert _route_after_build_allocation_plan(state) == "done"


def test_route_after_build_allocation_plan_generate_when_nonempty() -> None:
    state = _make_state(plan_slots=[_plan_slot(0)])
    assert _route_after_build_allocation_plan(state) == "generate"


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
        "generate_incoming_request",
        "generate_agent_turn",
        "generate_customer_turn",
        "validate_conversation",
        "commit_dialogue",
        "_route_consistency_or_skip",
        "_mark_cap_hit",
        "_bump_retry",
        "_mark_exhausted",
        "_mark_validation_skipped",
    }
    assert expected.issubset(node_names)


def _slot_with_drop(drop_after: int | None) -> _PlanSlot:
    return _plan_slot(1).model_copy(
        update={
            "rounds": [
                _RoundSpec(sequence=1, count=2, end_mode="dropped", drop_after_turns=drop_after),
                _RoundSpec(sequence=2, count=2),
            ]
        }
    )


def test_planned_drop_lowers_the_cap_for_that_contact_only() -> None:
    slot = _slot_with_drop(4)
    first = _make_state(
        plan_slots=[slot], current_dialogue_turns=_dialogue_turns(4), dialogue_turn_cap=20
    )
    assert _route_after_agent_turn(first) == "cap"
    assert _route_after_customer_turn(first) == "cap"
    second = first.model_copy(update={"round_index": 1})
    assert _route_after_agent_turn(second) == "customer"


def test_mark_cap_hit_distinguishes_planned_drop() -> None:
    dropped = _make_state(
        plan_slots=[_slot_with_drop(4)], current_dialogue_turns=_dialogue_turns(4)
    )
    assert asyncio.run(_mark_cap_hit_node(dropped)) == {"dialogue_end_reason": "dropped"}
    runaway = _make_state(plan_slots=[_plan_slot(1)], current_dialogue_turns=_dialogue_turns(20))
    assert asyncio.run(_mark_cap_hit_node(runaway)) == {
        "dialogue_end_reason": "cap_hit",
        "last_quality_flag": "warning:turn_cap_hit",
    }
