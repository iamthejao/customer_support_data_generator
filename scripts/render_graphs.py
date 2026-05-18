"""Render mermaid diagrams for the LangGraph pipeline + phase subgraphs.

The graph builders bind ``factory``/``db``/``settings`` into node closures via
``functools.partial``, but rendering only inspects the topology — the bound
dependencies are never invoked. This script supplies minimal stubs so the
diagrams can be rendered without a populated prompts/ or data/ tree (handy
inside tests that run from a tmp dir).
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from csfd.agents.factory import AgentFactory
from csfd.graph.phase1_graph import build_phase1_subgraph
from csfd.graph.phase2_graph import build_phase2_subgraph
from csfd.graph.pipeline_graph import build_pipeline_graph
from csfd.prompts.registry import PromptHandle, PromptRegistry
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


def _stub_settings() -> AppSettings:
    return AppSettings(
        pipeline=PipelineConfig(version="diagram", run_seed=0, budget=BudgetConfig()),
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


def _stub_factory() -> AgentFactory:
    env = jinja2.Environment(autoescape=False)
    stub_prompt = PromptHandle(
        name="stub",
        template=env.from_string("stub"),
        source="stub",
        version="stub",
    )
    reg = PromptRegistry(root=Path("/dev/null"))
    reg._handles = {
        "phase1.problem_brainstorm_v2": stub_prompt,
        "phase1.problem_combined_check": stub_prompt,
        "phase2.resolution_generator": stub_prompt,
        "phase2.resolution_combined_check": stub_prompt,
    }

    def _builder(_cfg: AgentLLMConfig) -> object:
        return object()  # never invoked during rendering

    return AgentFactory(
        prompts=reg,
        llm_builder=_builder,  # type: ignore[arg-type]
        agent_configs={
            "generator": AgentLLMConfig(provider="anthropic", model="stub"),
            "combined_checker": AgentLLMConfig(provider="anthropic", model="stub"),
        },
    )


def render_all(out_dir: Path) -> dict[str, Path]:
    """Render mermaid diagrams for the parent graph and both phase subgraphs.

    Returns a mapping of graph name → path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    settings = _stub_settings()
    factory = _stub_factory()
    db = Database(path=Path(":memory:"))

    parent = build_pipeline_graph(factory=factory, db=db, settings=settings, checkpointer=None)
    p_parent = out_dir / "pipeline.mmd"
    p_parent.write_text(parent.get_graph().draw_mermaid(), encoding="utf-8")
    written["pipeline"] = p_parent

    sub1 = build_phase1_subgraph(factory=factory, db=db, settings=settings)
    p1 = out_dir / "phase1.mmd"
    p1.write_text(sub1.get_graph().draw_mermaid(), encoding="utf-8")
    written["phase1"] = p1

    sub2 = build_phase2_subgraph(factory=factory, db=db, settings=settings)
    p2 = out_dir / "phase2.mmd"
    p2.write_text(sub2.get_graph().draw_mermaid(), encoding="utf-8")
    written["phase2"] = p2

    return written
