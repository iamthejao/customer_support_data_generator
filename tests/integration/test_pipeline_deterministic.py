"""End-to-end test of the deterministic, proportion-based pipeline.

Uses a FakeChatModel with canned structured outputs so the test runs offline.
Validation is disabled so a single canned ResolutionOutput is accepted for every
ticket type, regardless of target turn count. We then assert that the pipeline
produced exactly the right number of incoming_requests / resolutions / lineage
rows, and that those rows respect the configured proportions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel

from csfd.agents.base import Issue, Verdict
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.pipeline import (
    ProblemBrainstormOutput,
    ResolutionOutput,
    ResolutionTurnOutput,
    run_pipeline,
)
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
from csfd.storage.v2_repository import (
    IncomingRequestRepo,
    LineageRepo,
    ProblemV2Repo,
    ResolutionRepo,
)
from csfd.ticket_types.definitions import ProblemComplexity


def _canned_problem(
    complexity: ProblemComplexity = ProblemComplexity.SIMPLE,
) -> ProblemBrainstormOutput:
    return ProblemBrainstormOutput(
        title="Customer cannot log in",
        summary="A returning customer cannot sign in to their account.",
        background="Account exists; password recently reset; SSO turned off.",
        category="auth",
        complexity=complexity,
        resolution_hints={
            "docs_request": "Point to password-reset docs.",
            "l1": "Walk through password reset.",
            "l2": "Inspect auth logs and reset session tokens.",
            "l3": "Investigate identity provider misconfiguration.",
        },
    )


def _canned_resolution() -> ResolutionOutput:
    return ResolutionOutput(
        subject="Cannot log in",
        body="Hi team, I cannot log in to my account.",
        turns=[
            ResolutionTurnOutput(
                speaker="customer",
                name="Customer",
                content="Hi team, I cannot log in to my account.",
            ),
            ResolutionTurnOutput(
                speaker="agent",
                name="Agent",
                content="Thanks for reaching out. Could you try resetting your password?",
            ),
        ],
        resolved=True,
    )


def _canned_verdict() -> Verdict:
    return Verdict(
        checker="fake",
        passed=True,
        issues=[Issue(severity="warning", location="", rule_violated="", explanation="ok")] and [],
    )


def _build_settings(*, tmp_path: Path, total: int = 10, problems: int = 4) -> AppSettings:
    sqlite_path = tmp_path / "runs.sqlite"
    return AppSettings(
        pipeline=PipelineConfig(version="test", run_seed=7, budget=BudgetConfig()),
        agents={},
        problem_database=ProblemDatabaseConfig(
            count=problems,
            complexity_proportions={"simple": 0.5, "medium": 0.25, "complex": 0.25},
        ),
        tickets=TicketsConfig(
            total=total,
            type_proportions={
                "docs_request": 0.5,
                "l1": 0.3,
                "l2": 0.1,
                "l3": 0.1,
            },
            assignment_strategy="complexity_weighted",
            turns_per_type={"docs_request": 2, "l1": 3, "l2": 5, "l3": 7},
            tier_proportions={"standard": 0.6, "premium": 0.3, "enterprise": 0.1},
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


def _build_fake_factory(tmp_path: Path) -> AgentFactory:
    fake = FakeChatModel(
        structured={
            ProblemBrainstormOutput: _canned_problem(),
            ResolutionOutput: _canned_resolution(),
            Verdict: _canned_verdict(),
        }
    )

    def _builder(_cfg: object) -> BaseChatModel:
        return fake

    dummy_cfg = AgentLLMConfig(provider="anthropic", model="fake")
    configs = {
        name: dummy_cfg
        for name in (
            "generator",
            "combined_checker",
            "problem_brainstorm",
            "combined_problem_check",
            "resolution_generator",
            "combined_resolution_check",
        )
    }
    # The factory needs a populated prompt registry rooted at the project's prompts/ dir.
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(prompts=reg, llm_builder=_builder, agent_configs=configs)


@pytest.fixture
def company() -> CompanyProfile:
    return CompanyProfile(
        name="Acme", raw_markdown="# Acme\n\nWidget maker.", sections={"about": "Acme"}
    )


@pytest.fixture
def scenarios() -> ScenarioCatalogue:
    return ScenarioCatalogue(
        scenarios=[Scenario(category="auth", title="Login issues", summary="Cannot sign in.")],
        source_path=Path("seeds/scenarios_seed.md"),
    )


def test_pipeline_produces_exact_proportions(
    tmp_path: Path, company: CompanyProfile, scenarios: ScenarioCatalogue
) -> None:
    settings = _build_settings(tmp_path=tmp_path, total=10, problems=4)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    factory = _build_fake_factory(tmp_path)

    run_id = asyncio.run(
        run_pipeline(
            settings=settings,
            factory=factory,
            db=db,
            company=company,
            scenarios=scenarios,
        )
    )

    # Problem Database has the right number of rows.
    pdb = ProblemV2Repo(db).list_for_run(run_id)
    assert len(pdb) == 4
    # Complexity proportions: 0.5/0.25/0.25 over 4 -> 2/1/1.
    counts: dict[str, int] = {}
    for p in pdb:
        counts[p.complexity] = counts.get(p.complexity, 0) + 1
    assert counts == {"simple": 2, "medium": 1, "complex": 1}

    # Total tickets match config.
    ir_rows = IncomingRequestRepo(db).list_for_run(run_id)
    assert len(ir_rows) == 10
    assert ResolutionRepo(db).count_for_run(run_id) == 10
    assert LineageRepo(db).count_for_run(run_id) == 10

    # Ticket-type counts match the configured proportions (0.5/0.3/0.1/0.1 over 10).
    type_counts: dict[str, int] = {}
    for r in ir_rows:
        type_counts[r["ticket_type"]] = type_counts.get(r["ticket_type"], 0) + 1
    assert type_counts == {"docs_request": 5, "l1": 3, "l2": 1, "l3": 1}

    # Tier counts match the configured proportions (0.6/0.3/0.1 over 10).
    tier_counts: dict[str, int] = {}
    for r in ir_rows:
        tier_counts[r["customer_tier"]] = tier_counts.get(r["customer_tier"], 0) + 1
    assert tier_counts == {"standard": 6, "premium": 3, "enterprise": 1}


def test_pipeline_is_deterministic_across_runs(
    tmp_path: Path, company: CompanyProfile, scenarios: ScenarioCatalogue
) -> None:
    # Two runs against fresh DBs with identical config should produce identical
    # allocation plans slot-by-slot. We compare lineage rows ordered by
    # slot_index — the run_id prefix on problem_id obviously differs across
    # runs, but the within-run index suffix is the byte-identical invariant.
    settings_a = _build_settings(tmp_path=tmp_path / "a", total=10, problems=4)
    settings_b = _build_settings(tmp_path=tmp_path / "b", total=10, problems=4)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    db_a = Database(path=Path(settings_a.storage.sqlite_path))
    db_b = Database(path=Path(settings_b.storage.sqlite_path))
    apply_migrations(db_a)
    apply_migrations(db_b)
    factory = _build_fake_factory(tmp_path)
    run_a = asyncio.run(
        run_pipeline(
            settings=settings_a, factory=factory, db=db_a, company=company, scenarios=scenarios
        )
    )
    run_b = asyncio.run(
        run_pipeline(
            settings=settings_b, factory=factory, db=db_b, company=company, scenarios=scenarios
        )
    )

    def _slot_tuples(db: Database, run_id: str) -> list[tuple[int, str, str, str, str]]:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT slot_index, problem_id, ticket_type, customer_tier, customer_tone "
                "FROM lineage WHERE run_id = ? ORDER BY slot_index",
                (run_id,),
            ).fetchall()
        return [
            (
                int(r["slot_index"]),
                # Strip the run_id-derived prefix; the deterministic part is the
                # zero-padded index suffix, which is stable across runs.
                r["problem_id"].rsplit(":", 1)[-1],
                r["ticket_type"],
                r["customer_tier"],
                r["customer_tone"],
            )
            for r in rows
        ]

    slots_a = _slot_tuples(db_a, run_a)
    slots_b = _slot_tuples(db_b, run_b)
    assert slots_a == slots_b
    # Sanity: the byte-identical invariant must apply to every slot, not just the set.
    assert len(slots_a) == 10
    # customer_name embeds slot_index + tier — also catches sequence drift.
    names_a = [
        r["customer_name"]
        for r in sorted(
            IncomingRequestRepo(db_a).list_for_run(run_a),
            key=lambda r: r["request_uid"],
        )
    ]
    names_b = [
        r["customer_name"]
        for r in sorted(
            IncomingRequestRepo(db_b).list_for_run(run_b),
            key=lambda r: r["request_uid"].split(":", 1)[1],
        )
    ]
    # Strip the run_id prefix from db_a's request_uid for comparison.
    names_a_stripped = [n for n in names_a]
    assert names_a_stripped == names_b


def test_run_record_captures_provenance(
    tmp_path: Path, company: CompanyProfile, scenarios: ScenarioCatalogue
) -> None:
    """The pipeline must stamp `git_sha`, `stats_json`, and write `agent_traces` rows."""
    import json

    from csfd.storage.repository import AgentTraceRepo, RunRepo

    settings = _build_settings(tmp_path=tmp_path, total=4, problems=2)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    factory = _build_fake_factory(tmp_path)
    run_id = asyncio.run(
        run_pipeline(
            settings=settings, factory=factory, db=db, company=company, scenarios=scenarios
        )
    )
    run = RunRepo(db).get(run_id)
    # git_sha is captured when run from inside a git working tree; tests run in
    # the repo so this should be a 40-char hex string. Allow None when the
    # subprocess fails (e.g., on a CI sandbox without git) — we only assert it
    # is wired up to the helper.
    if run.git_sha is not None:
        assert len(run.git_sha) == 40
    assert run.stats_json is not None
    stats = json.loads(run.stats_json)
    assert stats["problem_count"] == 2
    assert stats["incoming_requests_count"] == 4
    assert stats["resolutions_count"] == 4
    assert stats["agent_traces_count"] >= 2  # at least one trace per problem
    # Validation is disabled in this fixture; quality_flag should reflect that.
    assert "warning:validation_skipped" in stats["quality_flags"]["problems"]
    # Trace rows exist with the expected prompt_id (sha-prefix of the rendered template).
    traces = AgentTraceRepo(db).list_for_run(run_id)
    assert len(traces) >= 2
    assert all(t.prompt_id for t in traces)
    assert all(t.model_provider == "anthropic" and t.model_id == "fake" for t in traces)


def test_lineage_links_every_ticket(
    tmp_path: Path, company: CompanyProfile, scenarios: ScenarioCatalogue
) -> None:
    settings = _build_settings(tmp_path=tmp_path, total=10, problems=4)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    factory = _build_fake_factory(tmp_path)
    run_id = asyncio.run(
        run_pipeline(
            settings=settings, factory=factory, db=db, company=company, scenarios=scenarios
        )
    )
    # Every lineage row must have non-NULL incoming_request_id and resolution_id.
    with db.connect() as conn:
        unlinked = conn.execute(
            "SELECT COUNT(*) AS n FROM lineage "
            "WHERE run_id = ? AND (incoming_request_id IS NULL OR resolution_id IS NULL)",
            (run_id,),
        ).fetchone()["n"]
    assert unlinked == 0
