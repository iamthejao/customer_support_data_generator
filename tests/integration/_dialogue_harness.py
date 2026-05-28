"""Shared harness for the Phase 2 turn-based dialogue integration tests.

Builds a real SQLite-backed Phase 2 subgraph wired to a FakeChatModel whose
structured outputs are scripted per test. Each test seeds one committed
problem and a single-slot allocation plan (tickets.total == 1), invokes the
subgraph once, and inspects the persisted rows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from csfd.agents.factory import AgentFactory
from csfd.graph.pipeline_graph import PipelineState
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import Scenario, ScenarioCatalogue
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
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import ProblemRecord, ProblemRepo, RunRecord, RunRepo

RUN_ID = "test-run"


def build_settings(
    tmp_path: Path, *, validation_enabled: bool, max_retries: int = 0, turn_cap: int = 20
) -> AppSettings:
    return AppSettings(
        pipeline=PipelineConfig(version="test", run_seed=7, budget=BudgetConfig()),
        agents={},
        problem_database=ProblemDatabaseConfig(count=1, complexity_proportions={"simple": 1.0}),
        tickets=TicketsConfig(
            total=1,
            type_proportions={"docs_request": 1.0, "l1": 0.0, "l2": 0.0, "l3": 0.0},
            assignment_strategy="complexity_weighted",
            dialogue=DialogueConfig(turn_cap=turn_cap),
            tier_proportions={"standard": 1.0},
            tone_proportions_per_type={
                "docs_request": {"neutral": 1.0},
                "l1": {"neutral": 1.0},
                "l2": {"neutral": 1.0},
                "l3": {"neutral": 1.0},
            },
        ),
        validation=ValidationConfig(enabled=validation_enabled, max_retries=max_retries),
        observability=ObservabilityConfig(),
        storage=StorageConfig(sqlite_path=str(tmp_path / "runs.sqlite")),
    )


def problem() -> ProblemRecord:
    return ProblemRecord(
        id="p001",
        run_id=RUN_ID,
        title="Power cycling",
        summary="PSU brownout under load.",
        background="Loose PSU connector intermittently drops voltage.",
        category="cooling",
        complexity="simple",
        resolution_hints={"docs_request": "Point to the reseat guide."},
        quality_flag=None,
        created_at=datetime.now(UTC),
        symptoms=["unit power-cycles every 20 minutes"],
        root_cause=["loose PSU connector"],
        fault_domain="hardware",
        customer_impact="degraded",
        tags=[],
    )


def factory(fake: FakeChatModel) -> AgentFactory:
    def _builder(_cfg: AgentLLMConfig) -> BaseChatModel:
        return fake

    dummy = AgentLLMConfig(provider="anthropic", model="fake")
    configs = {"generator": dummy, "combined_checker": dummy}
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(prompts=reg, llm_builder=_builder, agent_configs=configs)


def seed_run(db: Database) -> None:
    RunRepo(db).create(
        RunRecord(
            id=RUN_ID,
            phase="phase2",
            parent_run_id=None,
            status="running",
            started_at=datetime.now(UTC),
            completed_at=None,
            run_seed=7,
            pipeline_version="test",
            git_sha=None,
            config_snapshot_json=json.dumps({}),
            stats_json=None,
            error_summary=None,
        )
    )


def initial_state(settings: AppSettings) -> PipelineState:
    return PipelineState(
        run_id=RUN_ID,
        run_seed=settings.pipeline.run_seed or 0,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme", sections={}),
        scenarios=ScenarioCatalogue(
            scenarios=[Scenario(category="cooling", title="x", summary="x")],
            source_path=Path("seeds/scenarios_seed.md"),
        ),
        validation_enabled=settings.validation.enabled,
        max_retries=settings.validation.max_retries,
        problems_committed=[problem()],
    )


def setup_db(tmp_path: Path) -> Database:
    db = Database(path=tmp_path / "runs.sqlite")
    apply_migrations(db)
    seed_run(db)
    ProblemRepo(db).create(problem())
    return db


def resolution_row(db: Database) -> dict[str, Any]:
    with db.connect() as conn:
        r = conn.execute(
            "SELECT turns_json, turn_count, resolved, quality_flag "
            "FROM resolutions WHERE run_id = ?",
            (RUN_ID,),
        ).fetchone()
    return {
        "turns": json.loads(r["turns_json"]),
        "turn_count": int(r["turn_count"]),
        "resolved": bool(r["resolved"]),
        "quality_flag": r["quality_flag"],
    }


def trace_count(db: Database, *, node_name: str | None = None) -> int:
    with db.connect() as conn:
        if node_name is None:
            r = conn.execute(
                "SELECT COUNT(*) AS n FROM agent_traces WHERE run_id = ?", (RUN_ID,)
            ).fetchone()
        else:
            r = conn.execute(
                "SELECT COUNT(*) AS n FROM agent_traces WHERE run_id = ? AND node_name = ?",
                (RUN_ID, node_name),
            ).fetchone()
    return int(r["n"])
