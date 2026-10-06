"""The identifier registry in the pipeline: drawn before Phase 1, used and enforced by it.

With ``documents.enabled`` the run draws one registry entry per machine model
(``registry_writer`` names, code numbers), Phase 1 sees the parts and error
codes, a draft naming an unknown part number fails the code check without a
checker call, and a part the registry lacks is registered at commit. A run
that reuses an earlier run's problems takes that run's registry. Without
``documents.enabled`` nothing changes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.diagnosis import CandidateCause, DiagnosisPlan, DiagnosticCheck
from csfd.documents.registry import (
    NamedErrorCode,
    ProductFacts,
    RegistryNamesOutput,
    build_registry,
)
from csfd.models.fake import FakeChatModel
from csfd.outcomes import ProblemState
from csfd.pipeline import ProblemBrainstormOutput, run_pipeline
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import AssetModel, CaseFactsCatalogue, CompanyProfile
from csfd.seeds.scenarios import Scenario, ScenarioCatalogue
from csfd.settings import (
    AgentLLMConfig,
    AppSettings,
    BudgetConfig,
    DialogueConfig,
    DocumentsConfig,
    ObservabilityConfig,
    PipelineConfig,
    ProblemDatabaseConfig,
    StorageConfig,
    TicketsConfig,
    ValidationConfig,
)
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.product_facts import ProductFactsRepo
from csfd.storage.repository import AgentTraceRepo, ProblemRepo
from csfd.ticket_types.definitions import ProblemComplexity

SEED = 1
NAMES = RegistryNamesOutput(
    parts=["Thermocouple, type S", "Door seal", "Fuse, 6.3 A"],
    error_codes=[NamedErrorCode(kind="error", meaning="Over-temperature", action="Switch off")],
    menus=["Service > Calibration > Silver test"],
)


def _company() -> CompanyProfile:
    return CompanyProfile(
        name="Kalvora Dental",
        raw_markdown="# Kalvora Dental",
        case_facts=CaseFactsCatalogue(
            assets=[
                AssetModel("CF-600", "Ceramic firing furnace", "CF600-####-??"),
                AssetModel("SX-1500", "Sintering furnace", "SX1500-###-??"),
            ]
        ),
    )


def _scenarios() -> ScenarioCatalogue:
    return ScenarioCatalogue(
        scenarios=[Scenario(category="firing", title="x", summary="x")],
        source_path=Path("seeds/kalvora/scenarios_seed.md"),
    )


def _settings(tmp_path: Path, *, enabled: bool) -> AppSettings:
    return AppSettings(
        pipeline=PipelineConfig(version="t", run_seed=SEED, budget=BudgetConfig()),
        agents={},
        problem_database=ProblemDatabaseConfig(
            count=1, complexity_proportions={"simple": 1.0, "medium": 0.0, "complex": 0.0}
        ),
        tickets=TicketsConfig(
            total=0,
            type_proportions={"docs_request": 1.0, "l1": 0.0, "l2": 0.0, "l3": 0.0},
            assignment_strategy="complexity_weighted",
            dialogue=DialogueConfig(turn_cap=20),
            tier_proportions={"standard": 1.0, "premium": 0.0, "enterprise": 0.0},
            tone_proportions_per_type={
                t: {"neutral": 1.0} for t in ("docs_request", "l1", "l2", "l3")
            },
        ),
        validation=ValidationConfig(enabled=True, max_retries=3),
        observability=ObservabilityConfig(),
        storage=StorageConfig(sqlite_path=str(tmp_path / "runs.sqlite")),
        documents=DocumentsConfig(enabled=enabled),
    )


def _expected_registry(settings: AppSettings) -> dict[str, ProductFacts]:
    return build_registry(
        _company(),
        seed=SEED,
        calendar_start=settings.tickets.calendar.start,
        names={"CF-600": NAMES, "SX-1500": NAMES},
    )


def _problem(
    parts: list[str], symptom: str = "The display shows an error"
) -> ProblemBrainstormOutput:
    return ProblemBrainstormOutput(
        title="Restorations come out over-fired",
        summary="Crowns come out glassy.",
        background="Our CF-600 over-fires every crown since Monday.",
        symptoms=[symptom],
        root_cause=["Thermocouple drift"],
        category="firing",
        complexity=ProblemComplexity.SIMPLE,
        diagnosis_plan=DiagnosisPlan(
            candidate_causes=[
                CandidateCause(
                    cause="Thermocouple drift",
                    is_root_cause=True,
                    resolution_steps=["Replace the thermocouple"],
                    parts=parts,
                ),
                CandidateCause(cause="Wrong program", resolution_steps=["Load the program"]),
            ],
            checks=[
                DiagnosticCheck(
                    check="Silver test",
                    how_to_check="Run the silver test",
                    finding="The silver melts early",
                    rules_out=["Wrong program"],
                    confirms_cause=True,
                )
            ],
        ),
        viable_outcomes=[ProblemState.PENDING_PART],
    )


def _factory(fake: FakeChatModel) -> AgentFactory:
    def _builder(_cfg: AgentLLMConfig) -> BaseChatModel:
        return fake

    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    dummy = AgentLLMConfig(provider="anthropic", model="fake")
    return AgentFactory(
        prompts=reg,
        llm_builder=_builder,
        agent_configs={"generator": dummy, "combined_checker": dummy},
    )


def _run(
    settings: AppSettings, db: Database, fake: FakeChatModel, problems_from: str | None = None
) -> str:
    return asyncio.run(
        run_pipeline(
            settings=settings,
            factory=_factory(fake),
            db=db,
            company=_company(),
            scenarios=_scenarios(),
            problems_from=problems_from,
        )
    )


def test_registry_feeds_phase1_and_unknown_part_numbers_are_retried(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    expected = _expected_registry(settings)
    cf, sx = expected["CF-600"], expected["SX-1500"]
    thermocouple, code = cf.parts[0], cf.error_codes[0]

    fake = FakeChatModel(
        structured={Verdict: Verdict(checker="fake", passed=True, issues=[])},
        structured_seq={
            RegistryNamesOutput: [NAMES, NAMES],
            ProblemBrainstormOutput: [
                # A part of another machine: fails the code check.
                _problem([sx.parts[0].part_number]),
                _problem(
                    [thermocouple.part_number, "Heating element, 230 V"],
                    symptom=f"The display shows {code.code}",
                ),
            ],
        },
    )
    run_id = _run(settings, db, fake)

    stored = ProductFactsRepo(db).list_for_run(run_id)
    assert list(stored) == ["CF-600", "SX-1500"]
    assert stored["SX-1500"] == sx
    new_part = stored["CF-600"].parts[-1]
    assert new_part.name == "Heating element, 230 V"
    assert stored["CF-600"].parts[:-1] == cf.parts

    [problem] = ProblemRepo(db).list_for_run(run_id)
    assert problem.quality_flag is None
    assert problem.diagnosis_plan is not None
    root = problem.diagnosis_plan["candidate_causes"][0]
    assert root["parts"] == [thermocouple.entry, new_part.entry]

    traces = AgentTraceRepo(db).list_for_run(run_id)
    assert sorted(t.artifact_id or "" for t in traces if t.artifact_type == "registry") == [
        "CF-600",
        "SX-1500",
    ]
    problem_traces = [t for t in traces if t.artifact_type == "problem"]
    generator_inputs = [
        json.loads(t.input_json) for t in problem_traces if t.agent_role == "generator"
    ]
    assert len(generator_inputs) == 2
    seen = generator_inputs[0]["registry"]
    assert seen[0]["parts"][0] == {
        "part_number": thermocouple.part_number,
        "name": thermocouple.name,
    }
    # The first draft failed the code check, so the checker ran once, on the second.
    assert [t.attempt for t in problem_traces if t.agent_role == "checker"] == [1]


def test_reused_problems_keep_the_parent_registry(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    expected = _expected_registry(settings)
    fake = FakeChatModel(
        structured={
            Verdict: Verdict(checker="fake", passed=True, issues=[]),
            ProblemBrainstormOutput: _problem([expected["CF-600"].parts[0].part_number]),
        },
        # Only the parent run may call the registry writer.
        structured_seq={RegistryNamesOutput: [NAMES, NAMES]},
    )
    parent = _run(settings, db, fake)
    child = _run(settings, db, fake, problems_from=parent)
    repo = ProductFactsRepo(db)
    assert repo.list_for_run(child) == repo.list_for_run(parent)


def test_disabled_documents_draw_no_registry(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=False)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    # No canned RegistryNamesOutput: a registry call would fail the run.
    fake = FakeChatModel(
        structured={
            Verdict: Verdict(checker="fake", passed=True, issues=[]),
            ProblemBrainstormOutput: _problem(["KD-99-0000 Made-up part"]),
        }
    )
    run_id = _run(settings, db, fake)
    assert ProductFactsRepo(db).list_for_run(run_id) == {}
    [problem] = ProblemRepo(db).list_for_run(run_id)
    assert problem.diagnosis_plan is not None
    assert problem.diagnosis_plan["candidate_causes"][0]["parts"] == ["KD-99-0000 Made-up part"]
    traces = AgentTraceRepo(db).list_for_run(run_id)
    generator = next(t for t in traces if t.agent_role == "generator")
    assert "registry" not in json.loads(generator.input_json)
