"""CSFD CLI — Typer commands."""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path

import typer

from csfd.agents.factory import AgentFactory
from csfd.graph.compose import run_phase1, run_phase2
from csfd.models.registry import build_llm
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import parse_company_seed
from csfd.seeds.scenarios import parse_scenarios_seed
from csfd.settings import load_settings
from csfd.storage.db import Database
from csfd.storage.exporters import export_run_to_jsonl, export_run_to_parquet
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRepo,
    ProblemRepo,
    RunRepo,
)

app = typer.Typer(help="Customer-Service Fake Data — synthetic CS ticket generator.")
phase1_app = typer.Typer(help="Phase 1 (KB generation) commands.")
phase2_app = typer.Typer(help="Phase 2 (case generation) commands.")
app.add_typer(phase1_app, name="phase1")
app.add_typer(phase2_app, name="phase2")


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
"""


_SCENARIOS_SEED_STUB = """\
# Scenarios seed

Replace this stub with the scenario catalogue: categories, situational hints,
ticket-type expectations.
"""


@app.command()
def init() -> None:
    """Scaffold seeds/, data/, and an .env.example template in the current dir."""
    Path("seeds").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/exports").mkdir(exist_ok=True)
    company = Path("seeds/company_seed.md")
    if not company.exists():
        company.write_text(_COMPANY_SEED_STUB, encoding="utf-8")
    scenarios = Path("seeds/scenarios_seed.md")
    if not scenarios.exists():
        scenarios.write_text(_SCENARIOS_SEED_STUB, encoding="utf-8")
    env = Path(".env.example")
    if not env.exists():
        env.write_text(_DEFAULT_ENV, encoding="utf-8")
    typer.echo("Initialized seeds/, data/, .env.example.")


@app.command("db-migrate")
def db_migrate(
    sqlite_path: str = typer.Option(
        "data/runs.sqlite",
        "--sqlite-path",
        help="Path to SQLite database",
    ),
) -> None:
    """Apply pending SQL migrations to the runs database."""
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(path=path)
    apply_migrations(db)
    typer.echo(f"Applied migrations to {path}.")


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
    seeds_dir: str,
    company_seed: str | None,
    scenarios_seed: str | None,
) -> tuple[Path, Path]:
    company = Path(company_seed) if company_seed else Path(seeds_dir) / "company_seed.md"
    scenarios = Path(scenarios_seed) if scenarios_seed else Path(seeds_dir) / "scenarios_seed.md"
    return company, scenarios


@phase1_app.command("run")
def phase1_run(
    profile: str | None = typer.Option(None, "--profile"),
    seed: int | None = typer.Option(None, "--seed"),
    problems: int | None = typer.Option(None, "--problems"),
    coverage_rate: float | None = typer.Option(None, "--coverage-rate"),
    seeds_dir: str = typer.Option("seeds", "--seeds-dir"),
    company_seed: str | None = typer.Option(None, "--company-seed"),
    scenarios_seed: str | None = typer.Option(None, "--scenarios-seed"),
) -> None:
    """Run Phase 1 (KB generation) end-to-end."""
    settings = load_settings(profile=profile)
    if problems is not None:
        settings.phase1.problem_count = problems
    if coverage_rate is not None:
        settings.phase1.kb_coverage_target_rate = coverage_rate
    run_seed = seed if seed is not None else secrets.randbits(31)
    factory = _build_factory(profile)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    company_path, scenarios_path = _resolve_seed_paths(
        seeds_dir,
        company_seed,
        scenarios_seed,
    )
    company = parse_company_seed(company_path)
    scenarios = parse_scenarios_seed(scenarios_path)
    run_id = asyncio.run(
        run_phase1(
            factory=factory,
            db=db,
            phase1_cfg=settings.phase1,
            run_seed=run_seed,
            max_retries=settings.pipeline.budget.max_retries_per_artifact,
            company=company,
            scenarios=scenarios,
            pipeline_version=settings.pipeline.version,
        )
    )
    typer.echo(run_id)


@phase2_app.command("run")
def phase2_run(
    profile: str | None = typer.Option(None, "--profile"),
    seed: int | None = typer.Option(None, "--seed"),
    parent_run_id: str = typer.Option(..., "--parent-run-id"),
    seeds_dir: str = typer.Option("seeds", "--seeds-dir"),
    company_seed: str | None = typer.Option(None, "--company-seed"),
) -> None:
    """Run Phase 2 (case generation) chained off a prior Phase 1 run."""
    settings = load_settings(profile=profile)
    run_seed = seed if seed is not None else secrets.randbits(31)
    factory = _build_factory(profile)
    db = Database(path=Path(settings.storage.sqlite_path))
    company_path = Path(company_seed) if company_seed else Path(seeds_dir) / "company_seed.md"
    company = parse_company_seed(company_path)
    run_id = asyncio.run(
        run_phase2(
            factory=factory,
            db=db,
            phase2_cfg=settings.phase2,
            run_seed=run_seed,
            max_retries=settings.pipeline.budget.max_retries_per_artifact,
            parent_run_id=parent_run_id,
            company=company,
            pipeline_version=settings.pipeline.version,
        )
    )
    typer.echo(run_id)


@app.command()
def generate(
    profile: str | None = typer.Option(None, "--profile"),
    seed: int | None = typer.Option(None, "--seed"),
    problems: int | None = typer.Option(None, "--problems"),
    seeds_dir: str = typer.Option("seeds", "--seeds-dir"),
    company_seed: str | None = typer.Option(None, "--company-seed"),
    scenarios_seed: str | None = typer.Option(None, "--scenarios-seed"),
) -> None:
    """Run both Phase 1 and Phase 2 sequentially; print both run ids."""
    settings = load_settings(profile=profile)
    if problems is not None:
        settings.phase1.problem_count = problems
    run_seed = seed if seed is not None else secrets.randbits(31)
    factory = _build_factory(profile)
    db = Database(path=Path(settings.storage.sqlite_path))
    apply_migrations(db)
    company_path, scenarios_path = _resolve_seed_paths(
        seeds_dir,
        company_seed,
        scenarios_seed,
    )
    company = parse_company_seed(company_path)
    scenarios = parse_scenarios_seed(scenarios_path)

    async def _both() -> tuple[str, str]:
        p1 = await run_phase1(
            factory=factory,
            db=db,
            phase1_cfg=settings.phase1,
            run_seed=run_seed,
            max_retries=settings.pipeline.budget.max_retries_per_artifact,
            company=company,
            scenarios=scenarios,
            pipeline_version=settings.pipeline.version,
        )
        p2 = await run_phase2(
            factory=factory,
            db=db,
            phase2_cfg=settings.phase2,
            run_seed=run_seed,
            max_retries=settings.pipeline.budget.max_retries_per_artifact,
            parent_run_id=p1,
            company=company,
            pipeline_version=settings.pipeline.version,
        )
        return p1, p2

    p1, p2 = asyncio.run(_both())
    typer.echo(p1)
    typer.echo(p2)


@app.command()
def export(
    run_id: str = typer.Argument(...),
    format: str = typer.Option(
        "jsonl",
        "--format",
        help="jsonl | parquet | both",
    ),
    sqlite_path: str = typer.Option("data/runs.sqlite", "--sqlite-path"),
    out: str = typer.Option("data/exports", "--out"),
) -> None:
    """Export a run's artifacts to JSONL and/or Parquet under <out>/<run_id>/."""
    db = Database(path=Path(sqlite_path))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = {format} if format != "both" else {"jsonl", "parquet"}
    if "jsonl" in formats:
        export_run_to_jsonl(db, run_id, out_dir=out_dir)
    if "parquet" in formats:
        export_run_to_parquet(db, run_id, out_dir=out_dir)
    typer.echo(f"Exported run {run_id} to {out_dir / run_id}.")


@app.command()
def inspect(
    run_id: str = typer.Argument(...),
    sqlite_path: str = typer.Option("data/runs.sqlite", "--sqlite-path"),
) -> None:
    """Print a summary of a run: counts per artifact type."""
    db = Database(path=Path(sqlite_path))
    run = RunRepo(db).get(run_id)
    problems = ProblemRepo(db).list_for_run(run_id)
    kb_repo = KBArticleRepo(db)
    kb_count = sum(1 for p in problems if kb_repo.get_by_problem(p.id) is not None)
    with db.connect() as conn:
        ticket_count = conn.execute(
            "SELECT COUNT(*) AS n FROM tickets WHERE run_id = ?",
            (run_id,),
        ).fetchone()["n"]
        turn_count = conn.execute(
            """SELECT COUNT(*) AS n FROM turns t
               JOIN tickets ti ON ti.id = t.ticket_id
               WHERE ti.run_id = ?""",
            (run_id,),
        ).fetchone()["n"]
    typer.echo(f"Run: {run.id} ({run.phase}, {run.status})")
    typer.echo(f"  problems:    {len(problems)}")
    typer.echo(f"  kb_articles: {kb_count}")
    typer.echo(f"  tickets:     {ticket_count}")
    typer.echo(f"  turns:       {turn_count}")


@app.command("render-graphs")
def render_graphs(
    out: str = typer.Option("docs/diagrams", "--out"),
) -> None:
    """Regenerate docs/diagrams/*.mmd from the structural subgraphs."""
    from scripts.render_graphs import render_all

    written = render_all(Path(out))
    for name, path in written.items():
        typer.echo(f"  {name} → {path}")
