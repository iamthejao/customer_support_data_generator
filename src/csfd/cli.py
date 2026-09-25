"""CSFD CLI — Typer commands."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import cast, get_args

import typer

from csfd.agents.factory import AgentFactory
from csfd.models.registry import build_llm
from csfd.observability.logging import configure_logging
from csfd.pipeline import run_pipeline
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import parse_company_seed
from csfd.seeds.scenarios import parse_scenarios_seed
from csfd.settings import (
    Channel,
    Disfluency,
    HeaderStyle,
    SeedsConfig,
    SpeakerStyle,
    TranscriptStyleConfig,
    load_settings,
)
from csfd.storage.db import Database
from csfd.storage.exporters import export_run_to_jsonl, export_run_to_parquet
from csfd.storage.migrations.runner import StaleSchemaError, apply_migrations, check_schema
from csfd.storage.repository import (
    IncomingRequestRepo,
    LineageRepo,
    ProblemRepo,
    ResolutionRepo,
    RunRepo,
)
from csfd.storage.transcripts import export_run_transcripts

app = typer.Typer(help="Customer-Service Fake Data — synthetic CS ticket generator.")


_DEFAULT_ENV = """\
# Anthropic API key — required when any agent uses provider=anthropic
ANTHROPIC_API_KEY=

# OpenAI-compatible local endpoint (Ollama / llama-server / vLLM) — optional
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_API_KEY=local

# LangSmith — optional, enables hosted tracing
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=csfd
"""


_COMPANY_SEED_STUB = """\
# Company seed

Replace this stub with your company background: products, policies, tone of voice,
customer segments, KB style conventions.

## Case facts

What each case's facts are drawn from (see the README's "Case facts").

### Assets

| Model | Description | Serial format |
|---|---|---|
| XY-100 | Replace with a product customers own | XY100-####-?? |

### Caller roles

- maintenance technician

### Site locales

- en_US
"""


_SCENARIOS_SEED_STUB = """\
# Scenarios seed

Replace this stub with the scenario catalogue: categories, situational hints,
ticket-type expectations.
"""


_INIT_COMPANY = "my_company"


@app.command()
def init() -> None:
    """Scaffold seeds/<company>/, data/, and an .env.example template in the current dir."""
    company_dir = Path("seeds") / _INIT_COMPANY
    company_dir.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/exports").mkdir(exist_ok=True)
    company = company_dir / "company_seed.md"
    if not company.exists():
        company.write_text(_COMPANY_SEED_STUB, encoding="utf-8")
    scenarios = company_dir / "scenarios_seed.md"
    if not scenarios.exists():
        scenarios.write_text(_SCENARIOS_SEED_STUB, encoding="utf-8")
    env = Path(".env.example")
    if not env.exists():
        env.write_text(_DEFAULT_ENV, encoding="utf-8")
    typer.echo(
        f"Initialized {company_dir}/, data/, .env.example. "
        f"Select the seed with --company {_INIT_COMPANY} (or seeds.company in YAML)."
    )


def _with_current_schema(step: Callable[[Database], None], db: Database) -> None:
    """Run a schema step, turning a stale database into a clean CLI error."""
    try:
        step(db)
    except StaleSchemaError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("db-migrate")
def db_migrate(
    sqlite_path: str = typer.Option(
        "data/runs.sqlite",
        "--sqlite-path",
        help="Path to SQLite database",
    ),
) -> None:
    """Create the runs database schema (idempotent; drops nothing)."""
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(path=path)
    _with_current_schema(apply_migrations, db)
    typer.echo(f"Created schema at {path}.")


def _build_factory(profile: str | None) -> AgentFactory:
    settings = load_settings(profile=profile)
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=build_llm,
        agent_configs=settings.agents,
    )


def _resolve_seed_paths(
    seeds: SeedsConfig,
    company_seed: str | None,
    scenarios_seed: str | None,
) -> tuple[Path, Path]:
    """Seed files of the configured company; explicit file paths win."""
    company = Path(company_seed) if company_seed else seeds.company_path()
    scenarios = Path(scenarios_seed) if scenarios_seed else seeds.scenarios_path()
    for path, flag in ((company, "--company-seed"), (scenarios, "--scenarios-seed")):
        if not path.exists():
            raise typer.BadParameter(
                f"seed file not found: {path}. Pick a company under {seeds.dir}/ with "
                f"--company, or pass the file with {flag}.",
                param_hint=flag,
            )
    return company, scenarios


def _choice(value: str, allowed: object, flag: str) -> str:
    """Validate a CLI string against the members of a ``Literal`` type."""
    options: tuple[str, ...] = get_args(allowed)
    if value not in options:
        raise typer.BadParameter(f"must be one of: {', '.join(options)}", param_hint=flag)
    return value


@app.command()
def generate(
    profile: str | None = typer.Option(None, "--profile"),
    seed: int | None = typer.Option(None, "--seed"),
    problems: int | None = typer.Option(None, "--problems"),
    tickets: int | None = typer.Option(None, "--tickets"),
    company: str | None = typer.Option(
        None,
        "--company",
        help="Company seed: the folder under --seeds-dir holding company_seed.md and "
        "scenarios_seed.md (shipped: kalvora, norrholt). Overrides seeds.company.",
    ),
    seeds_dir: str | None = typer.Option(
        None, "--seeds-dir", help="Folder of company seeds. Overrides seeds.dir."
    ),
    company_seed: str | None = typer.Option(None, "--company-seed"),
    scenarios_seed: str | None = typer.Option(None, "--scenarios-seed"),
    channel: str | None = typer.Option(
        None,
        "--channel",
        help="Conversation format: 'email' (written tickets) or 'phone' (call transcripts). "
        "Overrides tickets.channel.",
    ),
    disfluency: str | None = typer.Option(
        None,
        "--disfluency",
        help="Phone speech style: none | light | moderate. Overrides tickets.phone.disfluency.",
    ),
    rounds: int | None = typer.Option(
        None,
        "--rounds",
        min=1,
        help="Contacts per case: every case becomes N related calls/emails (callbacks), "
        "with the first N-1 ending unresolved. Sets tickets.rounds.proportions to {N: 1.0}; "
        "configure a mix (e.g. {1: 0.6, 2: 0.3, 3: 0.1}) in YAML instead.",
    ),
) -> None:
    """Run the deterministic, proportion-based pipeline end-to-end.

    Generates the Problem Database (Phase 1) and then the full set of incoming
    requests + resolutions according to the configured proportions (Phase 2).
    Prints the single run id covering both phases.
    """
    channel_choice = cast(Channel, _choice(channel, Channel, "--channel")) if channel else None
    disfluency_choice = (
        cast(Disfluency, _choice(disfluency, Disfluency, "--disfluency")) if disfluency else None
    )
    settings = load_settings(profile=profile)
    configure_logging(json_output=settings.observability.structlog_json)
    # CLI overrides on top of YAML.
    if problems is not None:
        settings.problem_database.count = problems
    if tickets is not None:
        settings.tickets.total = tickets
    if seed is not None:
        settings.pipeline.run_seed = seed
    if channel_choice is not None:
        settings.tickets.channel = channel_choice
    if disfluency_choice is not None:
        settings.tickets.phone.disfluency = disfluency_choice
    if rounds is not None:
        settings.tickets.rounds.proportions = {rounds: 1.0}
    if company is not None:
        settings.seeds.company = company
    if seeds_dir is not None:
        settings.seeds.dir = seeds_dir
    factory = _build_factory(profile)
    db = Database(path=Path(settings.storage.sqlite_path))
    _with_current_schema(apply_migrations, db)
    company_path, scenarios_path = _resolve_seed_paths(
        settings.seeds,
        company_seed,
        scenarios_seed,
    )
    run_id = asyncio.run(
        run_pipeline(
            settings=settings,
            factory=factory,
            db=db,
            company=parse_company_seed(company_path),
            scenarios=parse_scenarios_seed(scenarios_path),
        )
    )
    typer.echo(run_id)


def _transcript_style(profile: str | None) -> TranscriptStyleConfig:
    """Transcript layout from YAML; the defaults when exporting outside a configured checkout."""
    if profile is None and not Path("config/default.yaml").exists():
        return TranscriptStyleConfig()
    return load_settings(profile=profile).storage.transcripts


@app.command()
def export(
    run_id: str = typer.Argument(...),
    format: str = typer.Option(
        "jsonl",
        "--format",
        help="jsonl | parquet | both (jsonl+parquet) | transcripts | all. "
        "'transcripts' writes plain-text call/email transcripts grouped by case "
        "under <out>/<run_id>/transcripts/.",
    ),
    sqlite_path: str = typer.Option("data/runs.sqlite", "--sqlite-path"),
    out: str = typer.Option("data/exports", "--out"),
    timestamps: bool = typer.Option(
        True,
        "--timestamps/--no-timestamps",
        help="Prefix phone transcript lines with [HH:MM:SS] call offsets.",
    ),
    speaker_style: str | None = typer.Option(
        None,
        "--speaker-style",
        help="Call transcript speaker labels: upper (AGENT:) | title (Agent:) | role "
        "(Agent: / Caller (<job role>):). Emails are unaffected. "
        "Overrides storage.transcripts.speaker_style.",
    ),
    header_style: str | None = typer.Option(
        None,
        "--header-style",
        help="Transcript header: csfd (full key: value header) | wissant (call_id / call_date "
        "block closed by ---) | none. Overrides storage.transcripts.header_style.",
    ),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    """Export a run's artifacts to JSONL, Parquet, and/or text transcripts under <out>/<run_id>/."""
    expansions = {
        "jsonl": {"jsonl"},
        "parquet": {"parquet"},
        "transcripts": {"transcripts"},
        "both": {"jsonl", "parquet"},
        "all": {"jsonl", "parquet", "transcripts"},
    }
    if format not in expansions:
        raise typer.BadParameter(f"must be one of: {', '.join(expansions)}", param_hint="--format")
    formats = expansions[format]
    style = _transcript_style(profile)
    if speaker_style is not None:
        style.speaker_style = cast(
            SpeakerStyle, _choice(speaker_style, SpeakerStyle, "--speaker-style")
        )
    if header_style is not None:
        style.header_style = cast(HeaderStyle, _choice(header_style, HeaderStyle, "--header-style"))
    db = Database(path=Path(sqlite_path))
    _with_current_schema(check_schema, db)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "jsonl" in formats:
        export_run_to_jsonl(db, run_id, out_dir=out_dir)
    if "parquet" in formats:
        export_run_to_parquet(db, run_id, out_dir=out_dir)
    if "transcripts" in formats:
        export_run_transcripts(
            db,
            run_id,
            out_dir=out_dir,
            timestamps=timestamps,
            speaker_style=style.speaker_style,
            header_style=style.header_style,
        )
    typer.echo(f"Exported run {run_id} to {out_dir / run_id}.")


@app.command()
def inspect(
    run_id: str = typer.Argument(...),
    sqlite_path: str = typer.Option("data/runs.sqlite", "--sqlite-path"),
) -> None:
    """Print a summary of a run: counts per artifact type."""
    db = Database(path=Path(sqlite_path))
    run = RunRepo(db).get(run_id)
    typer.echo(f"Run: {run.id} ({run.phase}, {run.status})")
    pdb = ProblemRepo(db).list_for_run(run_id)
    ir = IncomingRequestRepo(db).count_for_run(run_id)
    res = ResolutionRepo(db).count_for_run(run_id)
    lin = LineageRepo(db).count_for_run(run_id)
    typer.echo("Deterministic pipeline datasets:")
    typer.echo(f"  problem_database:  {len(pdb)}")
    typer.echo(f"  incoming_requests: {ir}")
    typer.echo(f"  resolutions:       {res}  (one per contact)")
    typer.echo(f"  lineage:           {lin}  (one per case)")


@app.command("render-graphs")
def render_graphs(
    out: str = typer.Option("docs/diagrams", "--out"),
) -> None:
    """Regenerate docs/diagrams/*.mmd from the structural subgraphs."""
    from csfd.graph.rendering import render_all

    written = render_all(Path(out))
    for name, path in written.items():
        typer.echo(f"  {name} → {path}")
