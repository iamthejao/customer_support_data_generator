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
    DialogueTurnOutput,
    IncomingRequestOutput,
    ProblemBrainstormOutput,
    run_pipeline,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import Scenario, ScenarioCatalogue
from csfd.settings import (
    AgentLLMConfig,
    AppSettings,
    BudgetConfig,
    Channel,
    DialogueConfig,
    ObservabilityConfig,
    PipelineConfig,
    ProblemDatabaseConfig,
    RoundsConfig,
    StorageConfig,
    TicketsConfig,
    ValidationConfig,
)
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    IncomingRequestRepo,
    LineageRepo,
    ProblemRepo,
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


def _canned_incoming() -> IncomingRequestOutput:
    return IncomingRequestOutput(
        subject="Cannot log in",
        body="Hi team, I cannot log in to my account.",
    )


def _canned_turn() -> DialogueTurnOutput:
    # done=True so each dialogue ends after the first agent turn (2 turns total).
    return DialogueTurnOutput(
        speaker="agent",
        content="Thanks for reaching out. Resetting your password should fix this.",
        done=True,
        done_reason="resolved",
    )


def _canned_verdict() -> Verdict:
    return Verdict(
        checker="fake",
        passed=True,
        issues=[Issue(severity="warning", location="", rule_violated="", explanation="ok")] and [],
    )


def _build_settings(
    *,
    tmp_path: Path,
    total: int = 10,
    problems: int = 4,
    channel: Channel = "email",
    rounds: RoundsConfig | None = None,
) -> AppSettings:
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
            dialogue=DialogueConfig(turn_cap=20),
            tier_proportions={"standard": 0.6, "premium": 0.3, "enterprise": 0.1},
            tone_proportions_per_type={
                "docs_request": {"neutral": 1.0},
                "l1": {"neutral": 1.0},
                "l2": {"neutral": 1.0},
                "l3": {"neutral": 1.0},
            },
            channel=channel,
            rounds=rounds or RoundsConfig(),
        ),
        validation=ValidationConfig(enabled=False, max_retries=0),
        observability=ObservabilityConfig(),
        storage=StorageConfig(sqlite_path=str(sqlite_path)),
    )


def _build_fake_factory(tmp_path: Path) -> AgentFactory:
    fake = FakeChatModel(
        structured={
            ProblemBrainstormOutput: _canned_problem(),
            IncomingRequestOutput: _canned_incoming(),
            DialogueTurnOutput: _canned_turn(),
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
            "incoming_request_generator",
            "customer_turn_generator",
            "agent_turn_generator",
            "conversation_consistency_check",
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
    pdb = ProblemRepo(db).list_for_run(run_id)
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
    def _strip_run(uid: str) -> str:
        # request_uid format: "<run_id>:<NNNNNN>:req". Strip the run_id prefix
        # so byte-identical comparison across runs is meaningful.
        return uid.split(":", 1)[1]

    names_a = [
        r["customer_name"]
        for r in sorted(
            IncomingRequestRepo(db_a).list_for_run(run_a),
            key=lambda r: _strip_run(r["request_uid"]),
        )
    ]
    names_b = [
        r["customer_name"]
        for r in sorted(
            IncomingRequestRepo(db_b).list_for_run(run_b),
            key=lambda r: _strip_run(r["request_uid"]),
        )
    ]
    assert names_a == names_b


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


def test_pipeline_handles_zero_tickets(
    tmp_path: Path, company: CompanyProfile, scenarios: ScenarioCatalogue
) -> None:
    """tickets.total == 0 must not crash; phase 2 is a no-op."""
    settings = _build_settings(tmp_path=tmp_path, total=0, problems=2)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    factory = _build_fake_factory(tmp_path)
    run_id = asyncio.run(
        run_pipeline(
            settings=settings, factory=factory, db=db, company=company, scenarios=scenarios
        )
    )
    assert IncomingRequestRepo(db).count_for_run(run_id) == 0
    assert ResolutionRepo(db).count_for_run(run_id) == 0
    assert LineageRepo(db).count_for_run(run_id) == 0


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


def test_phone_call_metadata_is_deterministic_across_runs(
    tmp_path: Path, company: CompanyProfile, scenarios: ScenarioCatalogue
) -> None:
    """Call start times and scripted greetings are pinned by seed + slot, not by the clock."""
    import json

    def _run(sub: str) -> list[tuple[str, str, str, str]]:
        (tmp_path / sub).mkdir()
        settings = _build_settings(tmp_path=tmp_path / sub, total=6, problems=2, channel="phone")
        db = Database(path=Path(settings.storage.sqlite_path))
        apply_migrations(db)
        run_id = asyncio.run(
            run_pipeline(
                settings=settings,
                factory=_build_fake_factory(tmp_path),
                db=db,
                company=company,
                scenarios=scenarios,
            )
        )
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT resolution_uid, channel, started_at, turns_json FROM resolutions "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return sorted(
            (
                r["resolution_uid"].split(":", 1)[1],
                r["channel"],
                r["started_at"],
                json.loads(r["turns_json"])[0]["content"],
            )
            for r in rows
        )

    a, b = _run("a"), _run("b")
    assert a == b
    assert len(a) == 6
    assert {row[1] for row in a} == {"phone"}
    assert len({row[2] for row in a}) == 6


def test_multi_contact_plan_is_deterministic_and_counted_per_case(
    tmp_path: Path, company: CompanyProfile, scenarios: ScenarioCatalogue
) -> None:
    """Contacts per case, their order, and their start times are pinned by seed + config."""
    import json

    from csfd.storage.repository import RunRepo

    rounds = RoundsConfig(proportions={1: 0.5, 2: 0.25, 3: 0.25})

    def _run(sub: str) -> tuple[list[tuple[str, int, int, str]], dict[str, object]]:
        (tmp_path / sub).mkdir()
        settings = _build_settings(
            tmp_path=tmp_path / sub, total=8, problems=2, channel="phone", rounds=rounds
        )
        db = Database(path=Path(settings.storage.sqlite_path))
        apply_migrations(db)
        run_id = asyncio.run(
            run_pipeline(
                settings=settings,
                factory=_build_fake_factory(tmp_path),
                db=db,
                company=company,
                scenarios=scenarios,
            )
        )
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT case_uid, round_index, round_count, started_at FROM resolutions "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        stats = json.loads(RunRepo(db).get(run_id).stats_json or "{}")
        contacts = sorted(
            (r["case_uid"].split(":", 1)[1], r["round_index"], r["round_count"], r["started_at"])
            for r in rows
        )
        return contacts, stats

    (a, stats), (b, _) = _run("a"), _run("b")
    assert a == b
    # 8 cases: 4 x 1 contact, 2 x 2, 2 x 3 -> 14 contacts.
    assert len(a) == 14
    assert len({case for case, *_ in a}) == 8
    for case in {case for case, *_ in a}:
        seqs = [(i, n, t) for c, i, n, t in a if c == case]
        assert [i for i, _, _ in seqs] == list(range(1, seqs[0][1] + 1))
        starts = [t for *_, t in seqs]
        assert starts == sorted(starts)
    assert stats["contacts_per_case"] == {"1": 4, "2": 2, "3": 2}
    assert stats["resolutions_count"] == 14
    assert stats["channel_counts"] == {"phone": 14}
    # Case-level breakdowns are not inflated by callbacks.
    assert sum(stats["type_counts"].values()) == 8  # type: ignore[attr-defined]
