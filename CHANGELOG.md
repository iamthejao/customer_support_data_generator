# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-15

### Added
- Two-phase synthetic customer-service ticket pipeline.
  - **Phase 1**: problem brainstorming, 3-checker parallelization (Consistency,
    Background, Scenario), coverage decider with target-rate balancing, KB
    article generation, retry-with-feedback loop bounded by `max_retries`.
  - **Phase 2**: ticket sampling with hard `has_kb=False → L3` routing, per-turn
    writer, probabilistic creative-noise injection, 3-checker per-turn validation,
    retry-with-feedback loop, multi-turn ticket assembly.
- 5 generic agent roles — Generator, Consistency, Background, Scenario,
  CreativeNoise — bound to specific (prompt, schema, llm) instances at graph
  nodes.
- LangGraph state machines with `Send`-based fan-out and `Annotated[list, operator.add]`
  fan-in reducers; content-hashed Jinja2 prompt registry.
- Pluggable LLM providers — Anthropic (`ChatAnthropic`) and OpenAI-compatible
  endpoints (`ChatOpenAI` → Ollama / llama-server / vLLM) — assignable per agent
  via YAML profiles.
- SQLite-based persistence with six application tables plus LangGraph's own
  checkpoint store.
- JSONL + Parquet exporters under `data/exports/<run_id>/`.
- CLI commands: `init`, `db-migrate`, `phase1 run`, `phase2 run`, `generate`,
  `export`, `inspect`, `render-graphs`.
- LangGraph Studio integration via `langgraph.json`.
- Apache-2.0 license.
- GitHub Actions CI on Python 3.12 / 3.13.
