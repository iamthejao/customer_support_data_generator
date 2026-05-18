"""Phase 1 retry path: first checker call fails, second passes, problem is accepted clean.

Validates the four properties the README promises:
  1. The retry counter advances (i.e. the second generator call happens).
  2. The accepted problem has no quality_flag (it passed cleanly on attempt 2).
  3. Two checker trace rows exist, the first verdict=fail, the second verdict=pass.
  4. Each checker trace has parent_trace_id pointing at the same-attempt generator trace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from csfd.agents.base import Issue, Verdict
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.pipeline import ProblemBrainstormOutput, ResolutionOutput, run_pipeline
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import Scenario, ScenarioCatalogue
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
from csfd.storage.repository import AgentTraceRepo
from csfd.storage.v2_repository import ProblemV2Repo
from csfd.ticket_types.definitions import ProblemComplexity


class RetryFake(FakeChatModel):
    """Verdict #1 fails, subsequent Verdicts pass. Problem + Resolution are canned."""

    verdict_calls: int = 0

    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel], **kw: Any
    ) -> Runnable[Any, BaseModel]:
        if schema is Verdict:

            def _verdict(_inputs: Any) -> BaseModel:
                self.verdict_calls += 1
                if self.verdict_calls == 1:
                    return Verdict(
                        checker="fake",
                        passed=False,
                        issues=[
                            Issue(
                                severity="error",
                                location="title",
                                rule_violated="too_short",
                                explanation="bad",
                            )
                        ],
                    )
                return Verdict(checker="fake", passed=True, issues=[])

            return RunnableLambda(_verdict)
        if schema is ProblemBrainstormOutput:
            return RunnableLambda(
                lambda _i: ProblemBrainstormOutput(
                    title="Login broken",
                    summary="x",
                    background="x",
                    category="auth",
                    complexity=ProblemComplexity.SIMPLE,
                    resolution_hints={
                        "docs_request": "x",
                        "l1": "x",
                        "l2": "x",
                        "l3": "x",
                    },
                )
            )
        if schema is ResolutionOutput:
            return RunnableLambda(
                lambda _i: ResolutionOutput(subject="s", body="b", turns=[], resolved=True)
            )
        raise KeyError(schema)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        pipeline=PipelineConfig(version="t", run_seed=1, budget=BudgetConfig()),
        agents={},
        problem_database=ProblemDatabaseConfig(
            count=1,
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
        validation=ValidationConfig(enabled=True, max_retries=3),
        observability=ObservabilityConfig(),
        storage=StorageConfig(sqlite_path=str(tmp_path / "runs.sqlite")),
    )


@pytest.fixture
def company() -> CompanyProfile:
    return CompanyProfile(name="Acme", raw_markdown="# Acme", sections={})


@pytest.fixture
def scenarios() -> ScenarioCatalogue:
    return ScenarioCatalogue(
        scenarios=[Scenario(category="auth", title="x", summary="x")],
        source_path=Path("seeds/scenarios_seed.md"),
    )


def test_phase1_retry_path_accepts_clean_on_second_attempt(
    tmp_path: Path, company: CompanyProfile, scenarios: ScenarioCatalogue
) -> None:
    fake = RetryFake()

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
    factory = AgentFactory(prompts=reg, llm_builder=_builder, agent_configs=configs)

    settings = _settings(tmp_path)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    run_id = asyncio.run(
        run_pipeline(
            settings=settings,
            factory=factory,
            db=db,
            company=company,
            scenarios=scenarios,
        )
    )

    # 1. Retry counter advanced: two checker calls happened.
    assert fake.verdict_calls == 2

    # 2. Accepted problem is flagged clean (passed on attempt 2, not retries_exhausted).
    pdb = ProblemV2Repo(db).list_for_run(run_id)
    assert len(pdb) == 1
    assert pdb[0].quality_flag is None

    # 3 & 4. Trace structure: 2 generator + 2 checker rows for the same problem,
    #        each checker linked to its same-attempt generator.
    traces = AgentTraceRepo(db).list_for_run(run_id)
    problem_traces = [t for t in traces if t.artifact_type == "problem"]
    by_role_attempt = {(t.agent_role, t.attempt): t for t in problem_traces}
    assert ("generator", 0) in by_role_attempt and ("checker", 0) in by_role_attempt
    assert ("generator", 1) in by_role_attempt and ("checker", 1) in by_role_attempt
    assert by_role_attempt[("checker", 0)].parent_trace_id == by_role_attempt[("generator", 0)].id
    assert by_role_attempt[("checker", 1)].parent_trace_id == by_role_attempt[("generator", 1)].id
    assert by_role_attempt[("checker", 0)].verdict == "fail"
    assert by_role_attempt[("checker", 1)].verdict == "pass"
