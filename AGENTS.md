# Repository Guidelines

## Project Structure & Module Organization

This is a Python package named `csfd` using a `src/` layout; code lives in `src/csfd/` and the CLI in `src/csfd/cli.py`. The LangGraph pipeline is in `src/csfd/graph/`: `pipeline_graph.py` (parent + `PipelineState`), `phase1_graph.py` (Problem Database), `phase2_graph.py` (dialogues). Supporting documents (IR, DOCX and Typst renderers, export, back-fill) live in `src/csfd/documents/`; the PDF template is `src/csfd/documents/templates/document.typ`. Prompt templates (Jinja) are in `prompts/`, company seeds in `seeds/<company>/`, configuration in `config/` (`default.yaml` plus `profiles/`), and the SQLite schema in `src/csfd/storage/migrations/schema.sql` (not upgraded in place: recreate `data/runs.sqlite` after a schema change). `README.md` owns the user-facing behaviour. Tests are split into `tests/unit/` and `tests/integration/`.

## Build, Test, and Development Commands

- `uv sync`: install runtime and development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest -q --tb=short`: run the full suite.
- `uv run pytest tests/unit`: run unit tests for faster feedback.
- `uv run ruff check src tests`: lint Python code.
- `uv run ruff format src tests`: format Python code with the project formatter.
- `uv run mypy src tests`: run strict type checking.
- `uv run csfd --help` (and `csfd <command> --help`): list commands and options.
- `uv run csfd generate --profile dev --seed 42`: small real run. It calls the configured models; only `dev` disables embedding dedup, which otherwise needs a local Ollama.

## Coding Style & Naming Conventions

Target Python 3.12+. Ruff enforces linting, import ordering, and formatting with a 100-character line length and double quotes. Mypy is strict, so keep public functions typed. Use `snake_case` for modules, functions, variables, and tests; use `PascalCase` for classes and Pydantic models. Keep first-party imports under `csfd`.

Prefer robust, well-known frameworks and libraries over building functionality in-house where one fits well. When several viable options exist for a design choice, do not pick one unilaterally: present the options with a recommendation and wait for the maintainer's decision.

## Testing Guidelines

Tests use `pytest` with `pytest-asyncio` enabled automatically. Name files `test_*.py` and place narrow behavior tests in `tests/unit/`; reserve `tests/integration/` for CLI, graph, persistence, and end-to-end flows. Live LLM tests must be marked `live_llm` and are not expected on normal PR checks. Prefer deterministic seeds and fake models.

## Commit & Pull Request Guidelines

Git history uses Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, and `ci:`. Keep commit messages imperative and scoped to one change. Pull requests should include a short description, linked issue or rationale, test evidence, and screenshots only when CLI or Studio output changes visually. Before opening a PR, run `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run mypy src tests`, and `uv run pytest -q --tb=short`.

## Security & Configuration Tips

Copy `.env.example` to `.env` for local credentials, but never commit `.env` or generated data. Treat `data/` outputs as local artifacts unless a review explicitly asks for fixtures.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
