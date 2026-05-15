# CSFD Foundation Implementation Plan (Plan 1 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the foundation layer of the `csfd` (customer-service-fake-data) multi-agent system — project skeleton, settings, errors, utilities, storage (SQLite + migrations + repos + exporters), seed parsers, prompt registry, LLM provider wiring, budget tracker, and observability — all under test, with zero agent or LangGraph code yet. After this plan, the test suite is green, the database can be migrated, all repos round-trip data, and the LLM provider abstraction returns mocked structured outputs.

**Architecture:** Vertical foundation slices, each TDD'd with fakes. Storage is SQLite with hand-rolled migrations and idempotent upserts. Settings are Pydantic v2 + YAML overlay (default ← profile ← env). LLM providers wrap LangChain `ChatAnthropic` / `ChatOpenAI` and use `with_structured_output`. Logging is `structlog` with context vars. Everything is type-checked under `mypy --strict`.

**Tech Stack:** Python 3.12+, uv, ruff, mypy strict, pytest, Pydantic v2, LangChain (anthropic + openai), `langgraph-checkpoint-sqlite`, structlog, tenacity, pybreaker, pyyaml, jinja2, pyarrow.

**Working directory:** `/Users/joao.augusto/Documents/Customer Service Fake Data` (the project root). All paths below are relative to this root.

---

## File Structure (Plan 1 scope)

| Path | Responsibility |
|---|---|
| `pyproject.toml` | uv-managed deps; tool configs (ruff, mypy, pytest) |
| `.python-version` | pin Python 3.12 |
| `.gitignore`, `.env.example`, `.pre-commit-config.yaml` | repo hygiene |
| `README.md` | placeholder; full README in Plan 5 |
| `src/csfd/__init__.py` | package marker |
| `src/csfd/errors.py` | typed exception hierarchy |
| `src/csfd/settings.py` | Pydantic Settings (env + YAML + profile merge) |
| `src/csfd/utils/hashing.py` | content hashing for prompts + dedup |
| `src/csfd/utils/rng.py` | seeded Random helpers |
| `src/csfd/utils/retry.py` | tenacity wrappers |
| `src/csfd/observability/logging.py` | structlog config |
| `src/csfd/observability/langsmith.py` | opt-in LangSmith env wiring |
| `src/csfd/storage/db.py` | SQLite connection manager |
| `src/csfd/storage/migrations/001_runs.sql` | runs table |
| `src/csfd/storage/migrations/002_problems_and_kb.sql` | problems + kb_articles tables |
| `src/csfd/storage/migrations/003_tickets_and_turns.sql` | tickets + turns tables |
| `src/csfd/storage/migrations/004_agent_traces.sql` | agent_traces table |
| `src/csfd/storage/migrations/__init__.py` + `runner.py` | migration runner |
| `src/csfd/storage/repository.py` | Run/Problem/KBArticle/Ticket/Turn/AgentTrace repos |
| `src/csfd/storage/exporters.py` | JSONL + Parquet exporters |
| `src/csfd/seeds/company.py` | parse company_seed.md → CompanyProfile |
| `src/csfd/seeds/scenarios.py` | parse scenarios_seed.md → ScenarioCatalogue |
| `src/csfd/ticket_types/definitions.py` | TicketType enum + per-type metadata |
| `src/csfd/prompts/registry.py` | PromptHandle + content-hash registry |
| `src/csfd/models/registry.py` | build_llm(agent_cfg) factory |
| `src/csfd/budget/tracker.py` | per-run token + USD budget |
| `src/csfd/budget/breaker.py` | pybreaker circuit breaker |
| `config/default.yaml` | full default config |
| `config/profiles/{dev,claude-only,local-only,mixed}.yaml` | profile overlays |
| `prompts/.gitkeep` (placeholders for phase1/phase2 subdirs) | structure only |
| `tests/conftest.py`, `tests/unit/test_*.py` | unit tests (one file per module) |
| `tests/fixtures/tiny_company_seed.md`, `tiny_scenarios_seed.md` | seed fixtures |

Out of scope in this plan (deferred to Plans 2–5): `agents/`, `phases/`, `graph/`, `cli.py`, `langgraph.json`, mermaid renderer, full README, CI workflows, integration cassettes.

---

## Task 1: Bootstrap project with `uv` and git

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`, `README.md`

- [ ] **Step 1: Initialize git and create `.python-version`**

```bash
cd "/Users/joao.augusto/Documents/Customer Service Fake Data"
git init -b main
echo "3.12" > .python-version
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
*.egg-info/
dist/
build/

# environments
.venv/
.env

# data (runtime artifacts)
data/

# editor / OS
.idea/
.vscode/
.DS_Store

# uv
# (uv.lock IS committed)
```

- [ ] **Step 3: Create `.env.example`**

```dotenv
# Anthropic
ANTHROPIC_API_KEY=

# OpenAI-compatible local endpoint (Ollama / llama.cpp / vLLM)
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_API_KEY=local

# LangSmith (optional)
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=csfd
```

- [ ] **Step 4: Create `pyproject.toml`**

```toml
[project]
name = "csfd"
version = "0.1.0"
description = "Customer Service Fake Data — synthetic CS ticket generator for benchmarking"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "Apache-2.0" }
authors = [{ name = "Joao Augusto" }]

dependencies = [
    "pydantic~=2.7",
    "pydantic-settings~=2.3",
    "pyyaml~=6.0",
    "jinja2~=3.1",
    "structlog~=24.1",
    "tenacity~=8.3",
    "pybreaker~=1.2",
    "pyarrow~=16.1",
    "typer~=0.12",
    "rich~=13.7",
    "langchain-core~=0.2",
    "langchain-anthropic~=0.1",
    "langchain-openai~=0.1",
    "langgraph~=0.1",
    "langgraph-checkpoint-sqlite~=1.0",
]

[project.scripts]
csfd = "csfd.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/csfd"]

[tool.uv]
dev-dependencies = [
    "pytest~=8.2",
    "pytest-recording~=0.13",
    "pytest-asyncio~=0.23",
    "ruff~=0.5",
    "mypy~=1.10",
    "types-pyyaml",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "RUF", "ASYNC"]
ignore = ["E501"]   # line length handled by formatter

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
disallow_untyped_decorators = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "live_llm: tests that hit real LLM APIs (gated; not run on PR)",
]
```

- [ ] **Step 5: Create placeholder `README.md`**

```markdown
# csfd — Customer Service Fake Data

Synthetic customer-service ticket generator for benchmarking. Multi-agent pipeline on LangGraph with Claude or local Llama models.

> 🚧 Under construction — full README and quickstart land in Plan 5.

See `docs/superpowers/specs/` for the design spec and `docs/superpowers/plans/` for the implementation plans.
```

- [ ] **Step 6: Sync deps and verify**

```bash
uv sync
uv run python -c "import pydantic, structlog, langgraph, langchain_anthropic; print('OK')"
```
Expected: `OK` printed; `uv.lock` created.

- [ ] **Step 7: Commit**

```bash
git add .python-version .gitignore .env.example pyproject.toml README.md uv.lock
git commit -m "chore: bootstrap project with uv and pyproject"
```

---

## Task 2: Configure tooling (ruff, mypy, pre-commit, pytest baseline)

**Files:**
- Create: `.pre-commit-config.yaml`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: check-merge-conflict
      - id: trailing-whitespace
      - id: end-of-file-fixer
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic~=2.7, types-pyyaml]
        args: [--strict, --no-warn-unused-ignores]
```

- [ ] **Step 2: Create `tests/__init__.py`** (empty) and **`tests/conftest.py`**

```python
# tests/conftest.py
import pytest


@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "test_runs.sqlite"
```

- [ ] **Step 3: Verify pytest discovers nothing yet (no tests written)**

```bash
uv run pytest -q
```
Expected: `no tests ran in 0.0Xs` — exits successfully.

- [ ] **Step 4: Verify ruff and mypy run clean on empty source**

```bash
uv run ruff check .
uv run mypy src tests || echo "expected: no source files yet"
```
Expected: ruff passes; mypy may report "no files" — acceptable for now.

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml tests/__init__.py tests/conftest.py
git commit -m "chore: add pre-commit, pytest, ruff, mypy configuration"
```

---

## Task 3: Create folder skeleton and config files

**Files:**
- Create: `src/csfd/__init__.py`, all `src/csfd/<subpkg>/__init__.py`, `config/default.yaml`, `config/profiles/*.yaml`, `seeds/.gitkeep`, `prompts/phase1/.gitkeep`, `prompts/phase2/.gitkeep`, `data/.gitkeep`, `docs/diagrams/.gitkeep`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `tests/fixtures/__init__.py`

- [ ] **Step 1: Create all package `__init__.py` files**

```bash
mkdir -p src/csfd/{utils,observability,storage/migrations,seeds,ticket_types,prompts,models,budget,agents,phases/phase1_kb,phases/phase2_cases,graph}
mkdir -p config/profiles seeds prompts/phase1 prompts/phase2 data docs/diagrams tests/unit tests/integration tests/fixtures
touch src/csfd/__init__.py
for d in utils observability storage storage/migrations seeds ticket_types prompts models budget agents phases phases/phase1_kb phases/phase2_cases graph; do
  touch "src/csfd/$d/__init__.py"
done
touch tests/unit/__init__.py tests/integration/__init__.py tests/fixtures/__init__.py
touch seeds/.gitkeep prompts/phase1/.gitkeep prompts/phase2/.gitkeep data/.gitkeep docs/diagrams/.gitkeep
```

- [ ] **Step 2: Create `config/default.yaml`**

```yaml
pipeline:
  version: "0.1.0"
  run_seed: null
  budget:
    max_tokens_per_run: 5000000
    max_usd_per_run: 10.0
    max_retries_per_artifact: 3

agents:
  generator:
    provider: anthropic
    model: claude-sonnet-4-7
    temperature: 0.85
    max_tokens: 4096
    timeout_s: 60
  consistency:
    provider: anthropic
    model: claude-haiku-4-5
    temperature: 0.1
    timeout_s: 30
  background:
    provider: anthropic
    model: claude-haiku-4-5
    temperature: 0.1
    timeout_s: 30
  scenario:
    provider: anthropic
    model: claude-haiku-4-5
    temperature: 0.1
    timeout_s: 30
  creative_noise:
    provider: anthropic
    model: claude-sonnet-4-7
    temperature: 0.95
    timeout_s: 45
  coverage_judge:
    provider: anthropic
    model: claude-haiku-4-5
    temperature: 0.2
    timeout_s: 30

phase1:
  problem_count: 100
  kb_coverage_target_rate: 0.7
  dedup_similarity_threshold: 0.88
  dedup_method: lexical

phase2:
  tickets_per_problem:
    has_kb: [3, 5]
    no_kb: [1, 2]
  ticket_type_weights:
    has_kb: { docs_request: 0.10, l1: 0.55, l2: 0.25, l3: 0.10 }
    no_kb: { l3: 1.0 }
  creative_noise_probability: 0.20
  noise_type_weights:
    typos_informal_phrasing: 0.25
    tangential_complaint: 0.15
    incomplete_info: 0.20
    agent_self_correction: 0.10
    sentiment_flip: 0.15
    wrong_kb_citation: 0.10
    code_switch_language: 0.05
  voting_policy: strict_all_pass
  retry_exhaustion_policy: commit_with_warning
  min_turns_per_ticket: 2
  max_turns_per_ticket: 12

observability:
  structlog_json: true
  langsmith_enabled: false
  langsmith_project: csfd
  persist_stream_updates: true

storage:
  sqlite_path: "data/runs.sqlite"
  exports_dir: "data/exports"
  exports_format: ["jsonl", "parquet"]
```

- [ ] **Step 3: Create profile overlays**

`config/profiles/dev.yaml`:
```yaml
phase1:
  problem_count: 5
phase2:
  tickets_per_problem:
    has_kb: [1, 2]
    no_kb: [1, 1]
pipeline:
  budget:
    max_tokens_per_run: 100000
    max_usd_per_run: 1.0
```

`config/profiles/claude-only.yaml`:
```yaml
# no overrides — default is already claude-only
```

`config/profiles/local-only.yaml`:
```yaml
agents:
  generator:
    provider: openai_compat
    model: llama3.3:70b
    base_url: http://localhost:11434/v1
  consistency:
    provider: openai_compat
    model: llama3.3:70b
    base_url: http://localhost:11434/v1
  background:
    provider: openai_compat
    model: llama3.3:70b
    base_url: http://localhost:11434/v1
  scenario:
    provider: openai_compat
    model: llama3.3:70b
    base_url: http://localhost:11434/v1
  creative_noise:
    provider: openai_compat
    model: llama3.3:70b
    base_url: http://localhost:11434/v1
  coverage_judge:
    provider: openai_compat
    model: llama3.3:70b
    base_url: http://localhost:11434/v1
```

`config/profiles/mixed.yaml`:
```yaml
agents:
  generator:
    provider: anthropic
    model: claude-sonnet-4-7
  consistency:
    provider: openai_compat
    model: llama3.1:8b
    base_url: http://localhost:11434/v1
  background:
    provider: openai_compat
    model: llama3.1:8b
    base_url: http://localhost:11434/v1
  scenario:
    provider: openai_compat
    model: llama3.1:8b
    base_url: http://localhost:11434/v1
```

- [ ] **Step 4: Verify directory tree**

```bash
find src config tests prompts seeds data -type f | sort
```
Expected: lists all `__init__.py`, all YAML files, `.gitkeep` files.

- [ ] **Step 5: Commit**

```bash
git add src config tests prompts seeds data docs/diagrams
git commit -m "chore: add folder skeleton and default config files"
```

---

## Task 4: `errors.py` — typed exception hierarchy

**Files:**
- Create: `src/csfd/errors.py`
- Test: `tests/unit/test_errors.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_errors.py`:
```python
import pytest
from pydantic import ValidationError as PydValidationError, BaseModel

from csfd.errors import (
    BudgetExceededError,
    ConvergenceError,
    PipelineError,
    SchemaValidationError,
    TransportError,
)


class _DummyModel(BaseModel):
    n: int


def test_hierarchy_roots_at_pipeline_error():
    for cls in (TransportError, SchemaValidationError, BudgetExceededError, ConvergenceError):
        assert issubclass(cls, PipelineError)


def test_schema_validation_error_carries_pydantic_errors_and_raw():
    try:
        _DummyModel.model_validate({"n": "not-an-int"})
    except PydValidationError as e:
        err = SchemaValidationError(errors=e.errors(), raw_output='{"n":"not-an-int"}')
    assert err.raw_output == '{"n":"not-an-int"}'
    assert len(err.errors) == 1
    assert "n" in err.errors[0]["loc"]


def test_budget_error_carries_stats():
    err = BudgetExceededError(stats={"tokens": 6_000_000, "usd": 12.5})
    assert err.stats["tokens"] == 6_000_000
    assert "tokens" in str(err)
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_errors.py -v
```
Expected: `ImportError: cannot import name 'PipelineError' from 'csfd.errors'`.

- [ ] **Step 3: Implement `src/csfd/errors.py`**

```python
"""Typed exception hierarchy for the csfd pipeline."""
from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base class for all csfd pipeline errors."""


class TransportError(PipelineError):
    """Network-level failure (5xx, timeout, connection reset). Retried at the call layer."""


class SchemaValidationError(PipelineError):
    """LLM output failed Pydantic validation. Routed back to the generator with feedback."""

    def __init__(self, *, errors: list[dict[str, Any]], raw_output: str) -> None:
        self.errors = errors
        self.raw_output = raw_output
        super().__init__(f"Schema validation failed with {len(errors)} error(s)")


class BudgetExceededError(PipelineError):
    """Token or USD budget hit. The run aborts; committed artifacts persist."""

    def __init__(self, *, stats: dict[str, Any]) -> None:
        self.stats = stats
        super().__init__(f"Budget exceeded: {stats}")


class ConvergenceError(PipelineError):
    """Verdict-issue list identical across two consecutive retries — generator is stuck."""
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_errors.py -v
```
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/errors.py tests/unit/test_errors.py
git commit -m "feat: add typed exception hierarchy for pipeline errors"
```

---

## Task 5: `utils/hashing.py` — content hashing

**Files:**
- Create: `src/csfd/utils/hashing.py`
- Test: `tests/unit/test_hashing.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_hashing.py`:
```python
from csfd.utils.hashing import short_hash, stable_json_hash


def test_short_hash_is_deterministic_and_12_chars():
    h1 = short_hash("hello world")
    h2 = short_hash("hello world")
    assert h1 == h2
    assert len(h1) == 12


def test_short_hash_strips_leading_trailing_whitespace():
    assert short_hash("hello world") == short_hash("   hello world   ")


def test_short_hash_changes_on_content_change():
    assert short_hash("a") != short_hash("b")


def test_stable_json_hash_ignores_dict_ordering():
    a = {"k1": 1, "k2": [3, 4]}
    b = {"k2": [3, 4], "k1": 1}
    assert stable_json_hash(a) == stable_json_hash(b)
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_hashing.py -v
```
Expected: `ModuleNotFoundError: No module named 'csfd.utils.hashing'`.

- [ ] **Step 3: Implement `src/csfd/utils/hashing.py`**

```python
"""Content hashing helpers used for prompt versioning and dedup."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def short_hash(content: str, length: int = 12) -> str:
    """Stable short SHA-256 hex digest of `content` after stripping whitespace."""
    digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
    return digest[:length]


def stable_json_hash(obj: Any, length: int = 12) -> str:
    """Hash an object via canonical JSON encoding (sorted keys, separators)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return short_hash(payload, length=length)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_hashing.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/utils/hashing.py tests/unit/test_hashing.py
git commit -m "feat: add content hashing helpers (short_hash, stable_json_hash)"
```

---

## Task 6: `utils/rng.py` — seeded RNG helpers

**Files:**
- Create: `src/csfd/utils/rng.py`
- Test: `tests/unit/test_rng.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_rng.py`:
```python
from csfd.utils.rng import derive_rng, weighted_choice


def test_derive_rng_is_deterministic_with_same_seed_and_label():
    a = derive_rng(seed=42, label="phase1")
    b = derive_rng(seed=42, label="phase1")
    assert a.random() == b.random()


def test_derive_rng_diverges_on_different_label():
    a = derive_rng(seed=42, label="phase1")
    b = derive_rng(seed=42, label="phase2")
    assert a.random() != b.random()


def test_weighted_choice_respects_weights_over_many_trials():
    rng = derive_rng(seed=123, label="weights")
    weights = {"a": 0.8, "b": 0.2}
    counts = {"a": 0, "b": 0}
    for _ in range(5000):
        counts[weighted_choice(weights, rng)] += 1
    # 'a' should dominate; loose bounds to avoid flakes
    assert counts["a"] > 3500
    assert counts["b"] < 1500


def test_weighted_choice_rejects_empty_weights():
    rng = derive_rng(seed=0, label="x")
    import pytest
    with pytest.raises(ValueError):
        weighted_choice({}, rng)
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_rng.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/utils/rng.py`**

```python
"""Seeded RNG helpers for reproducible sampling."""
from __future__ import annotations

import hashlib
from random import Random
from typing import Mapping


def derive_rng(seed: int, label: str) -> Random:
    """Derive a deterministic Random instance for a (seed, label) pair.

    Different labels under the same seed produce independent sub-streams.
    """
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    sub_seed = int.from_bytes(digest[:8], "big", signed=False)
    return Random(sub_seed)


def weighted_choice(weights: Mapping[str, float], rng: Random) -> str:
    """Pick a key proportional to its weight. Raises if weights is empty."""
    if not weights:
        raise ValueError("weights must be non-empty")
    keys = list(weights.keys())
    values = [weights[k] for k in keys]
    return rng.choices(keys, weights=values, k=1)[0]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_rng.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/utils/rng.py tests/unit/test_rng.py
git commit -m "feat: add seeded RNG helpers (derive_rng, weighted_choice)"
```

---

## Task 7: `utils/retry.py` — tenacity wrappers

**Files:**
- Create: `src/csfd/utils/retry.py`
- Test: `tests/unit/test_retry.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_retry.py`:
```python
import pytest

from csfd.errors import SchemaValidationError, TransportError
from csfd.utils.retry import with_transport_retry


@pytest.mark.asyncio
async def test_with_transport_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    @with_transport_retry(max_attempts=4, initial_seconds=0.0, max_seconds=0.0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransportError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_with_transport_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    @with_transport_retry(max_attempts=3, initial_seconds=0.0, max_seconds=0.0)
    async def always_fail() -> None:
        calls["n"] += 1
        raise TransportError("dead")

    with pytest.raises(TransportError):
        await always_fail()
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_with_transport_retry_does_not_retry_other_errors():
    calls = {"n": 0}

    @with_transport_retry(max_attempts=3, initial_seconds=0.0, max_seconds=0.0)
    async def wrong_error() -> None:
        calls["n"] += 1
        raise SchemaValidationError(errors=[], raw_output="")

    with pytest.raises(SchemaValidationError):
        await wrong_error()
    assert calls["n"] == 1
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_retry.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/utils/retry.py`**

```python
"""Tenacity-based retry wrappers for transport-layer errors."""
from __future__ import annotations

from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from csfd.errors import TransportError

T = TypeVar("T")


def with_transport_retry(
    *,
    max_attempts: int = 5,
    initial_seconds: float = 1.0,
    max_seconds: float = 30.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator: retry an async function on TransportError with exp-backoff + jitter."""

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_jitter(initial=initial_seconds, max=max_seconds),
                retry=retry_if_exception_type(TransportError),
                reraise=True,
            ):
                with attempt:
                    return await fn(*args, **kwargs)
            raise RuntimeError("unreachable")

        return wrapper

    return decorator
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_retry.py -v
```
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/utils/retry.py tests/unit/test_retry.py
git commit -m "feat: add tenacity-based transport retry decorator"
```

---

## Task 8: `observability/logging.py` — structlog configuration

**Files:**
- Create: `src/csfd/observability/logging.py`
- Test: `tests/unit/test_logging.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_logging.py`:
```python
import io
import json
import structlog

from csfd.observability.logging import bind_context, configure_logging


def test_configure_logging_emits_json_with_contextvars():
    buf = io.StringIO()
    configure_logging(json_output=True, stream=buf)
    bind_context(run_id="run-1", agent_role="generator")
    log = structlog.get_logger("test")
    log.info("hello", extra="value")
    payload = json.loads(buf.getvalue().splitlines()[-1])
    assert payload["event"] == "hello"
    assert payload["run_id"] == "run-1"
    assert payload["agent_role"] == "generator"
    assert payload["extra"] == "value"
    assert payload["level"] == "info"


def test_configure_logging_emits_kv_when_not_json():
    buf = io.StringIO()
    configure_logging(json_output=False, stream=buf)
    log = structlog.get_logger("test2")
    log.info("hi")
    out = buf.getvalue()
    assert "event=" in out or "hi" in out
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_logging.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/observability/logging.py`**

```python
"""structlog configuration with contextvars-based correlation."""
from __future__ import annotations

import logging
import sys
from typing import IO, Any

import structlog


def configure_logging(*, json_output: bool = True, stream: IO[str] | None = None) -> None:
    """Configure structlog + stdlib logging.

    Args:
        json_output: if True, emit JSON; else key=value.
        stream: where to write log lines (defaults to sys.stderr).
    """
    target = stream if stream is not None else sys.stderr

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(file=target),
        cache_logger_on_first_use=False,
    )
    # mirror stdlib root logger to the same target for libraries that use logging
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(target)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def bind_context(**kwargs: Any) -> None:
    """Bind key-value pairs into the contextvars-scoped log context for this task."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_logging.py -v
```
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/observability/logging.py tests/unit/test_logging.py
git commit -m "feat: add structlog configuration with contextvars binding"
```

---

## Task 9: `settings.py` — Pydantic Settings with YAML overlay

**Files:**
- Create: `src/csfd/settings.py`
- Test: `tests/unit/test_settings.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_settings.py`:
```python
import pytest

from csfd.settings import AppSettings, load_settings


def test_load_settings_with_default_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = load_settings(default_path="config/default.yaml", profile=None)
    assert s.pipeline.version == "0.1.0"
    assert s.phase1.problem_count == 100
    assert s.agents["generator"].provider == "anthropic"
    assert s.phase2.voting_policy == "strict_all_pass"


def test_load_settings_with_dev_profile_overrides_problem_count():
    s = load_settings(default_path="config/default.yaml", profile="dev")
    assert s.phase1.problem_count == 5
    assert s.pipeline.budget.max_usd_per_run == 1.0


def test_load_settings_with_local_only_profile_switches_providers():
    s = load_settings(default_path="config/default.yaml", profile="local-only")
    assert s.agents["generator"].provider == "openai_compat"
    assert "11434" in (s.agents["generator"].base_url or "")


def test_load_settings_unknown_profile_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(default_path="config/default.yaml", profile="does-not-exist")


def test_env_var_overrides_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    s = load_settings(default_path="config/default.yaml", profile=None)
    assert s.env.anthropic_api_key == "test-key"
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_settings.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/settings.py`**

```python
"""Pydantic Settings: YAML default + profile overlay + environment variables."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BudgetConfig(BaseModel):
    max_tokens_per_run: int = 5_000_000
    max_usd_per_run: float = 10.0
    max_retries_per_artifact: int = 3


class PipelineConfig(BaseModel):
    version: str = "0.1.0"
    run_seed: int | None = None
    budget: BudgetConfig = Field(default_factory=BudgetConfig)


class AgentLLMConfig(BaseModel):
    provider: Literal["anthropic", "openai_compat"]
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout_s: int = 60
    base_url: str | None = None
    api_key: str | None = None


class Phase1Config(BaseModel):
    problem_count: int = 100
    kb_coverage_target_rate: float = 0.7
    dedup_similarity_threshold: float = 0.88
    dedup_method: Literal["lexical", "embedding"] = "lexical"


class TicketsPerProblem(BaseModel):
    has_kb: tuple[int, int] = (3, 5)
    no_kb: tuple[int, int] = (1, 2)


class TicketTypeWeights(BaseModel):
    has_kb: dict[str, float]
    no_kb: dict[str, float]


class Phase2Config(BaseModel):
    tickets_per_problem: TicketsPerProblem = Field(default_factory=TicketsPerProblem)
    ticket_type_weights: TicketTypeWeights
    creative_noise_probability: float = 0.2
    noise_type_weights: dict[str, float] = Field(default_factory=dict)
    voting_policy: Literal["strict_all_pass", "quorum_2_of_3"] = "strict_all_pass"
    retry_exhaustion_policy: Literal["commit_with_warning", "skip"] = "commit_with_warning"
    min_turns_per_ticket: int = 2
    max_turns_per_ticket: int = 12


class ObservabilityConfig(BaseModel):
    structlog_json: bool = True
    langsmith_enabled: bool = False
    langsmith_project: str = "csfd"
    persist_stream_updates: bool = True


class StorageConfig(BaseModel):
    sqlite_path: str = "data/runs.sqlite"
    exports_dir: str = "data/exports"
    exports_format: list[str] = Field(default_factory=lambda: ["jsonl", "parquet"])


class EnvSecrets(BaseSettings):
    """Environment-sourced secrets and overrides."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    anthropic_api_key: str | None = None
    local_base_url: str | None = None
    local_api_key: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None


class AppSettings(BaseModel):
    pipeline: PipelineConfig
    agents: dict[str, AgentLLMConfig]
    phase1: Phase1Config
    phase2: Phase2Config
    observability: ObservabilityConfig
    storage: StorageConfig
    env: EnvSecrets = Field(default_factory=EnvSecrets)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(
    *, default_path: str = "config/default.yaml", profile: str | None = None
) -> AppSettings:
    with open(default_path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if profile:
        profile_path = Path("config/profiles") / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        with open(profile_path, encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        data = _deep_merge(data, overlay)

    return AppSettings(**data)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_settings.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/settings.py tests/unit/test_settings.py
git commit -m "feat: add Pydantic Settings with YAML default + profile overlay + env"
```

---

## Task 10: `storage/db.py` — SQLite connection manager

**Files:**
- Create: `src/csfd/storage/db.py`
- Test: `tests/unit/test_db.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_db.py`:
```python
from csfd.storage.db import Database


def test_database_creates_file_and_pragmas(tmp_db_path):
    db = Database(path=tmp_db_path)
    with db.connect() as conn:
        cur = conn.execute("PRAGMA foreign_keys")
        assert cur.fetchone()[0] == 1
        cur = conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0].lower() == "wal"
    assert tmp_db_path.exists()


def test_database_context_commits_on_success(tmp_db_path):
    db = Database(path=tmp_db_path)
    with db.connect() as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t (v) VALUES (?)", ("hello",))
    with db.connect() as conn:
        row = conn.execute("SELECT v FROM t").fetchone()
        assert row[0] == "hello"


def test_database_context_rolls_back_on_exception(tmp_db_path):
    db = Database(path=tmp_db_path)
    with db.connect() as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")

    import pytest
    with pytest.raises(RuntimeError):
        with db.connect() as conn:
            conn.execute("INSERT INTO t (id) VALUES (1)")
            raise RuntimeError("boom")

    with db.connect() as conn:
        rows = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert rows == 0
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_db.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/storage/db.py`**

```python
"""SQLite connection manager with WAL + foreign keys."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_db.py -v
```
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/db.py tests/unit/test_db.py
git commit -m "feat: add SQLite Database connection manager with WAL and FK"
```

---

## Task 11: Migration SQL files (001–004)

**Files:**
- Create: `src/csfd/storage/migrations/001_runs.sql`, `002_problems_and_kb.sql`, `003_tickets_and_turns.sql`, `004_agent_traces.sql`

- [ ] **Step 1: Create `001_runs.sql`**

```sql
CREATE TABLE IF NOT EXISTS runs (
    id                   TEXT PRIMARY KEY,
    phase                TEXT NOT NULL CHECK (phase IN ('phase1', 'phase2', 'full')),
    parent_run_id        TEXT REFERENCES runs(id),
    status               TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'aborted_budget')),
    started_at           TIMESTAMP NOT NULL,
    completed_at         TIMESTAMP,
    run_seed             INTEGER NOT NULL,
    pipeline_version     TEXT NOT NULL,
    git_sha              TEXT,
    config_snapshot_json TEXT NOT NULL,
    stats_json           TEXT,
    error_summary        TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_phase ON runs(phase);
CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_run_id);
```

- [ ] **Step 2: Create `002_problems_and_kb.sql`**

```sql
CREATE TABLE IF NOT EXISTS problems (
    id                     TEXT PRIMARY KEY,
    run_id                 TEXT NOT NULL REFERENCES runs(id),
    title                  TEXT NOT NULL,
    description            TEXT NOT NULL,
    category               TEXT NOT NULL,
    severity               TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    has_kb                 INTEGER NOT NULL CHECK (has_kb IN (0, 1)),
    coverage_reasoning     TEXT,
    coverage_confidence    TEXT CHECK (coverage_confidence IN ('low', 'medium', 'high')),
    metadata_json          TEXT,
    quality_flag           TEXT,
    unresolved_issues_json TEXT,
    created_at             TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_problems_run ON problems(run_id);
CREATE INDEX IF NOT EXISTS idx_problems_has_kb ON problems(has_kb);

CREATE TABLE IF NOT EXISTS kb_articles (
    id                         TEXT PRIMARY KEY,
    run_id                     TEXT NOT NULL REFERENCES runs(id),
    problem_id                 TEXT NOT NULL UNIQUE REFERENCES problems(id),
    title                      TEXT NOT NULL,
    content_markdown           TEXT NOT NULL,
    content_hash               TEXT NOT NULL,
    troubleshooting_steps_json TEXT NOT NULL,
    prerequisites_json         TEXT,
    metadata_json              TEXT,
    version                    INTEGER NOT NULL DEFAULT 1,
    quality_flag               TEXT,
    unresolved_issues_json     TEXT,
    created_at                 TIMESTAMP NOT NULL
);
```

- [ ] **Step 3: Create `003_tickets_and_turns.sql`**

```sql
CREATE TABLE IF NOT EXISTS tickets (
    id                     TEXT PRIMARY KEY,
    run_id                 TEXT NOT NULL REFERENCES runs(id),
    problem_id             TEXT NOT NULL REFERENCES problems(id),
    kb_article_id          TEXT REFERENCES kb_articles(id),
    ticket_type            TEXT NOT NULL CHECK (ticket_type IN ('docs_request', 'l1', 'l2', 'l3')),
    priority               TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    status                 TEXT NOT NULL CHECK (status IN ('resolved', 'unresolved', 'escalated')),
    subject                TEXT NOT NULL,
    customer_persona_json  TEXT NOT NULL,
    agent_persona_json     TEXT NOT NULL,
    ground_truth_json      TEXT,
    metadata_json          TEXT,
    quality_flag           TEXT,
    unresolved_issues_json TEXT,
    created_at             TIMESTAMP NOT NULL,
    resolved_at            TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tickets_run ON tickets(run_id);
CREATE INDEX IF NOT EXISTS idx_tickets_problem ON tickets(problem_id);
CREATE INDEX IF NOT EXISTS idx_tickets_type ON tickets(ticket_type);

CREATE TABLE IF NOT EXISTS turns (
    id                  TEXT PRIMARY KEY,
    ticket_id           TEXT NOT NULL REFERENCES tickets(id),
    turn_index          INTEGER NOT NULL,
    speaker             TEXT NOT NULL CHECK (speaker IN ('customer', 'agent', 'system')),
    speaker_persona     TEXT,
    content             TEXT NOT NULL,
    intent              TEXT,
    kb_references_json  TEXT,
    noise_applied       INTEGER NOT NULL DEFAULT 0 CHECK (noise_applied IN (0, 1)),
    noise_type          TEXT,
    quality_flag        TEXT,
    created_at          TIMESTAMP NOT NULL,
    UNIQUE (ticket_id, turn_index)
);
CREATE INDEX IF NOT EXISTS idx_turns_ticket ON turns(ticket_id);
```

- [ ] **Step 4: Create `004_agent_traces.sql`**

```sql
CREATE TABLE IF NOT EXISTS agent_traces (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(id),
    thread_id           TEXT NOT NULL,
    node_name           TEXT NOT NULL,
    agent_role          TEXT NOT NULL,
    artifact_type       TEXT NOT NULL,
    artifact_id         TEXT,
    attempt             INTEGER NOT NULL DEFAULT 0,
    prompt_id           TEXT NOT NULL,
    input_json          TEXT NOT NULL,
    output_json         TEXT,
    verdict             TEXT CHECK (verdict IN ('pass', 'fail')),
    verdict_issues_json TEXT,
    model_provider      TEXT NOT NULL CHECK (model_provider IN ('anthropic', 'openai_compat', 'fake')),
    model_id            TEXT NOT NULL,
    tokens_in           INTEGER,
    tokens_out          INTEGER,
    cost_usd_estimated  REAL,
    latency_ms          INTEGER,
    parent_trace_id     TEXT REFERENCES agent_traces(id),
    status              TEXT NOT NULL CHECK (status IN ('ok', 'transport_error', 'schema_error', 'budget_error')),
    error_class         TEXT,
    error_message       TEXT,
    created_at          TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_run ON agent_traces(run_id);
CREATE INDEX IF NOT EXISTS idx_traces_artifact ON agent_traces(artifact_type, artifact_id);
CREATE INDEX IF NOT EXISTS idx_traces_role ON agent_traces(agent_role);
CREATE INDEX IF NOT EXISTS idx_traces_prompt ON agent_traces(prompt_id);
```

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/migrations/*.sql
git commit -m "feat: add SQLite migration SQL for runs, problems, kb, tickets, turns, traces"
```

---

## Task 12: Migration runner

**Files:**
- Create: `src/csfd/storage/migrations/runner.py`
- Test: `tests/unit/test_migrations.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_migrations.py`:
```python
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations, applied_versions


def test_apply_migrations_creates_all_tables(tmp_db_path):
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    with db.connect() as conn:
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    for required in {"runs", "problems", "kb_articles", "tickets", "turns", "agent_traces", "schema_migrations"}:
        assert required in names


def test_apply_migrations_is_idempotent(tmp_db_path):
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    apply_migrations(db)  # second call must not raise
    assert set(applied_versions(db)) == {"001", "002", "003", "004"}
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_migrations.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/storage/migrations/runner.py`**

```python
"""Apply numbered .sql migrations idempotently."""
from __future__ import annotations

import re
from pathlib import Path

from csfd.storage.db import Database

_MIGRATIONS_DIR = Path(__file__).parent
_MIGRATION_PATTERN = re.compile(r"^(\d{3})_.*\.sql$")


def _discover() -> list[tuple[str, Path]]:
    files = []
    for p in sorted(_MIGRATIONS_DIR.iterdir()):
        m = _MIGRATION_PATTERN.match(p.name)
        if m:
            files.append((m.group(1), p))
    return files


def applied_versions(db: Database) -> list[str]:
    with db.connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  version TEXT PRIMARY KEY,"
            "  applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    return [r[0] for r in rows]


def apply_migrations(db: Database) -> list[str]:
    """Apply pending migrations; return list of newly-applied versions."""
    already = set(applied_versions(db))
    newly_applied: list[str] = []
    for version, path in _discover():
        if version in already:
            continue
        sql = path.read_text(encoding="utf-8")
        with db.connect() as conn:
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
        newly_applied.append(version)
    return newly_applied
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_migrations.py -v
```
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/migrations/runner.py tests/unit/test_migrations.py
git commit -m "feat: add idempotent SQL migration runner"
```

---

## Task 13: Repository — `RunRepo`

**Files:**
- Create: `src/csfd/storage/repository.py` (RunRepo only — other repos added in later tasks)
- Test: `tests/unit/test_repo_run.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_repo_run.py`:
```python
import json
from datetime import datetime, timezone
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import RunRecord, RunRepo


def _setup(tmp_db_path):
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    return RunRepo(db)


def test_create_and_get_run(tmp_db_path):
    repo = _setup(tmp_db_path)
    run = RunRecord(
        id=str(uuid4()),
        phase="phase1",
        parent_run_id=None,
        status="running",
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        run_seed=42,
        pipeline_version="0.1.0",
        git_sha="abc1234",
        config_snapshot_json=json.dumps({"k": "v"}),
        stats_json=None,
        error_summary=None,
    )
    repo.create(run)
    got = repo.get(run.id)
    assert got.id == run.id
    assert got.run_seed == 42
    assert got.status == "running"


def test_update_status_and_stats(tmp_db_path):
    repo = _setup(tmp_db_path)
    run_id = str(uuid4())
    repo.create(RunRecord(
        id=run_id, phase="phase2", parent_run_id=None, status="running",
        started_at=datetime.now(timezone.utc), completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None,
        config_snapshot_json="{}", stats_json=None, error_summary=None,
    ))
    repo.update_status(run_id, status="completed", stats={"tokens": 123})
    got = repo.get(run_id)
    assert got.status == "completed"
    assert json.loads(got.stats_json)["tokens"] == 123


def test_get_unknown_id_raises(tmp_db_path):
    import pytest
    repo = _setup(tmp_db_path)
    with pytest.raises(KeyError):
        repo.get("does-not-exist")
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_repo_run.py -v
```
Expected: `ModuleNotFoundError` or attribute errors.

- [ ] **Step 3: Implement `src/csfd/storage/repository.py`** (initial — RunRepo only)

```python
"""Data-access layer. Each *Repo encapsulates idempotent CRUD for one table."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from csfd.storage.db import Database


@dataclass(slots=True)
class RunRecord:
    id: str
    phase: str
    parent_run_id: str | None
    status: str
    started_at: datetime
    completed_at: datetime | None
    run_seed: int
    pipeline_version: str
    git_sha: str | None
    config_snapshot_json: str
    stats_json: str | None
    error_summary: str | None


class RunRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, run: RunRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO runs
                  (id, phase, parent_run_id, status, started_at, completed_at,
                   run_seed, pipeline_version, git_sha, config_snapshot_json,
                   stats_json, error_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id, run.phase, run.parent_run_id, run.status,
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                    run.run_seed, run.pipeline_version, run.git_sha,
                    run.config_snapshot_json, run.stats_json, run.error_summary,
                ),
            )

    def get(self, run_id: str) -> RunRecord:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return _row_to_run(row)

    def update_status(
        self,
        run_id: str,
        *,
        status: str,
        stats: dict[str, Any] | None = None,
        error_summary: str | None = None,
        completed: bool = True,
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, completed_at = ?, stats_json = COALESCE(?, stats_json),
                    error_summary = COALESCE(?, error_summary)
                WHERE id = ?
                """,
                (
                    status,
                    datetime.now(timezone.utc).isoformat() if completed else None,
                    json.dumps(stats) if stats is not None else None,
                    error_summary,
                    run_id,
                ),
            )


def _row_to_run(row: Any) -> RunRecord:
    return RunRecord(
        id=row["id"],
        phase=row["phase"],
        parent_run_id=row["parent_run_id"],
        status=row["status"],
        started_at=datetime.fromisoformat(row["started_at"]),
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        run_seed=row["run_seed"],
        pipeline_version=row["pipeline_version"],
        git_sha=row["git_sha"],
        config_snapshot_json=row["config_snapshot_json"],
        stats_json=row["stats_json"],
        error_summary=row["error_summary"],
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_repo_run.py -v
```
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/repository.py tests/unit/test_repo_run.py
git commit -m "feat: add RunRepo with idempotent create + status update"
```

---

## Task 14: Repository — `ProblemRepo`

**Files:**
- Modify: `src/csfd/storage/repository.py` (append `ProblemRepo`)
- Test: `tests/unit/test_repo_problem.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_repo_problem.py`:
```python
import json
from datetime import datetime, timezone
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import ProblemRecord, ProblemRepo, RunRecord, RunRepo


def _setup_run(tmp_db_path) -> tuple[Database, str]:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="running",
        started_at=datetime.now(timezone.utc), completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None, config_snapshot_json="{}",
        stats_json=None, error_summary=None,
    ))
    return db, run_id


def test_create_and_list_problems(tmp_db_path):
    db, run_id = _setup_run(tmp_db_path)
    repo = ProblemRepo(db)
    for i in range(3):
        repo.create(ProblemRecord(
            id=str(uuid4()), run_id=run_id,
            title=f"Issue {i}", description="desc",
            category="billing", severity="medium",
            has_kb=(i % 2 == 0), coverage_reasoning="r",
            coverage_confidence="medium",
            metadata_json=json.dumps({"tag": i}),
            quality_flag=None, unresolved_issues_json=None,
            created_at=datetime.now(timezone.utc),
        ))
    rows = repo.list_for_run(run_id)
    assert len(rows) == 3
    has_kb = repo.list_for_run(run_id, has_kb=True)
    assert len(has_kb) == 2


def test_create_is_idempotent_on_duplicate_id(tmp_db_path):
    db, run_id = _setup_run(tmp_db_path)
    repo = ProblemRepo(db)
    pid = str(uuid4())
    rec = ProblemRecord(
        id=pid, run_id=run_id, title="t", description="d",
        category="c", severity="low", has_kb=False,
        coverage_reasoning=None, coverage_confidence=None,
        metadata_json=None, quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(timezone.utc),
    )
    repo.create(rec)
    repo.create(rec)  # second insert must not raise
    assert len(repo.list_for_run(run_id)) == 1
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_repo_problem.py -v
```
Expected: `ImportError: cannot import name 'ProblemRepo'`.

- [ ] **Step 3: Append to `src/csfd/storage/repository.py`**

```python
@dataclass(slots=True)
class ProblemRecord:
    id: str
    run_id: str
    title: str
    description: str
    category: str
    severity: str
    has_kb: bool
    coverage_reasoning: str | None
    coverage_confidence: str | None
    metadata_json: str | None
    quality_flag: str | None
    unresolved_issues_json: str | None
    created_at: datetime


class ProblemRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, p: ProblemRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO problems
                  (id, run_id, title, description, category, severity, has_kb,
                   coverage_reasoning, coverage_confidence, metadata_json,
                   quality_flag, unresolved_issues_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p.id, p.run_id, p.title, p.description, p.category, p.severity,
                    1 if p.has_kb else 0,
                    p.coverage_reasoning, p.coverage_confidence, p.metadata_json,
                    p.quality_flag, p.unresolved_issues_json,
                    p.created_at.isoformat(),
                ),
            )

    def list_for_run(self, run_id: str, *, has_kb: bool | None = None) -> list[ProblemRecord]:
        sql = "SELECT * FROM problems WHERE run_id = ?"
        params: list[Any] = [run_id]
        if has_kb is not None:
            sql += " AND has_kb = ?"
            params.append(1 if has_kb else 0)
        sql += " ORDER BY created_at"
        with self.db.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            ProblemRecord(
                id=r["id"], run_id=r["run_id"], title=r["title"],
                description=r["description"], category=r["category"],
                severity=r["severity"], has_kb=bool(r["has_kb"]),
                coverage_reasoning=r["coverage_reasoning"],
                coverage_confidence=r["coverage_confidence"],
                metadata_json=r["metadata_json"],
                quality_flag=r["quality_flag"],
                unresolved_issues_json=r["unresolved_issues_json"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_repo_problem.py -v
```
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/repository.py tests/unit/test_repo_problem.py
git commit -m "feat: add ProblemRepo with idempotent create + filtered list"
```

---

## Task 15: Repository — `KBArticleRepo`

**Files:**
- Modify: `src/csfd/storage/repository.py`
- Test: `tests/unit/test_repo_kb.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_repo_kb.py`:
```python
import json
from datetime import datetime, timezone
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRecord, KBArticleRepo,
    ProblemRecord, ProblemRepo,
    RunRecord, RunRepo,
)


def _make_problem(db) -> str:
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="running",
        started_at=datetime.now(timezone.utc), completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None, config_snapshot_json="{}",
        stats_json=None, error_summary=None,
    ))
    pid = str(uuid4())
    ProblemRepo(db).create(ProblemRecord(
        id=pid, run_id=run_id, title="t", description="d", category="c",
        severity="low", has_kb=True, coverage_reasoning=None,
        coverage_confidence=None, metadata_json=None,
        quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(timezone.utc),
    ))
    return run_id, pid


def test_create_and_get_kb_article(tmp_db_path):
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id, pid = _make_problem(db)

    repo = KBArticleRepo(db)
    aid = str(uuid4())
    repo.create(KBArticleRecord(
        id=aid, run_id=run_id, problem_id=pid,
        title="How to reset", content_markdown="## Step 1\n...",
        content_hash="abcdef012345",
        troubleshooting_steps_json=json.dumps([{"step": "do X", "expected_result": "Y"}]),
        prerequisites_json=None, metadata_json=None,
        version=1, quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(timezone.utc),
    ))
    got = repo.get_by_problem(pid)
    assert got is not None
    assert got.id == aid
    assert got.title == "How to reset"


def test_get_by_problem_returns_none_when_absent(tmp_db_path):
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    _ , pid = _make_problem(db)
    repo = KBArticleRepo(db)
    assert repo.get_by_problem(pid) is None
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_repo_kb.py -v
```
Expected: `ImportError: cannot import name 'KBArticleRepo'`.

- [ ] **Step 3: Append to `src/csfd/storage/repository.py`**

```python
@dataclass(slots=True)
class KBArticleRecord:
    id: str
    run_id: str
    problem_id: str
    title: str
    content_markdown: str
    content_hash: str
    troubleshooting_steps_json: str
    prerequisites_json: str | None
    metadata_json: str | None
    version: int
    quality_flag: str | None
    unresolved_issues_json: str | None
    created_at: datetime


class KBArticleRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, a: KBArticleRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO kb_articles
                  (id, run_id, problem_id, title, content_markdown, content_hash,
                   troubleshooting_steps_json, prerequisites_json, metadata_json,
                   version, quality_flag, unresolved_issues_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a.id, a.run_id, a.problem_id, a.title, a.content_markdown,
                    a.content_hash, a.troubleshooting_steps_json,
                    a.prerequisites_json, a.metadata_json, a.version,
                    a.quality_flag, a.unresolved_issues_json,
                    a.created_at.isoformat(),
                ),
            )

    def get_by_problem(self, problem_id: str) -> KBArticleRecord | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM kb_articles WHERE problem_id = ?", (problem_id,)
            ).fetchone()
        if row is None:
            return None
        return KBArticleRecord(
            id=row["id"], run_id=row["run_id"], problem_id=row["problem_id"],
            title=row["title"], content_markdown=row["content_markdown"],
            content_hash=row["content_hash"],
            troubleshooting_steps_json=row["troubleshooting_steps_json"],
            prerequisites_json=row["prerequisites_json"],
            metadata_json=row["metadata_json"],
            version=row["version"],
            quality_flag=row["quality_flag"],
            unresolved_issues_json=row["unresolved_issues_json"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_repo_kb.py -v
```
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/repository.py tests/unit/test_repo_kb.py
git commit -m "feat: add KBArticleRepo with idempotent create + get_by_problem"
```

---

## Task 16: Repository — `TicketRepo` and `TurnRepo`

**Files:**
- Modify: `src/csfd/storage/repository.py`
- Test: `tests/unit/test_repo_ticket_turn.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_repo_ticket_turn.py`:
```python
import json
from datetime import datetime, timezone
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemRecord, ProblemRepo,
    RunRecord, RunRepo,
    TicketRecord, TicketRepo,
    TurnRecord, TurnRepo,
)


def _bootstrap(tmp_db_path) -> tuple[Database, str, str]:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase2", parent_run_id=None, status="running",
        started_at=datetime.now(timezone.utc), completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None, config_snapshot_json="{}",
        stats_json=None, error_summary=None,
    ))
    pid = str(uuid4())
    ProblemRepo(db).create(ProblemRecord(
        id=pid, run_id=run_id, title="t", description="d",
        category="c", severity="low", has_kb=False,
        coverage_reasoning=None, coverage_confidence=None,
        metadata_json=None, quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(timezone.utc),
    ))
    return db, run_id, pid


def test_create_ticket_and_turns(tmp_db_path):
    db, run_id, pid = _bootstrap(tmp_db_path)
    t_repo = TicketRepo(db)
    tu_repo = TurnRepo(db)

    tid = str(uuid4())
    t_repo.create(TicketRecord(
        id=tid, run_id=run_id, problem_id=pid, kb_article_id=None,
        ticket_type="l3", priority="high", status="resolved",
        subject="My printer is on fire",
        customer_persona_json=json.dumps({"name": "Alex"}),
        agent_persona_json=json.dumps({"name": "Tier3-Specialist"}),
        ground_truth_json=None, metadata_json=None,
        quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(timezone.utc), resolved_at=None,
    ))
    for i, (speaker, content) in enumerate([("customer", "help!"), ("agent", "describe?"), ("customer", "smoke")]):
        tu_repo.create(TurnRecord(
            id=str(uuid4()), ticket_id=tid, turn_index=i,
            speaker=speaker, speaker_persona=None, content=content,
            intent=None, kb_references_json=None,
            noise_applied=False, noise_type=None,
            quality_flag=None,
            created_at=datetime.now(timezone.utc),
        ))
    turns = tu_repo.list_for_ticket(tid)
    assert len(turns) == 3
    assert [t.turn_index for t in turns] == [0, 1, 2]


def test_turn_unique_index_per_ticket(tmp_db_path):
    db, run_id, pid = _bootstrap(tmp_db_path)
    t_repo = TicketRepo(db)
    tu_repo = TurnRepo(db)
    tid = str(uuid4())
    t_repo.create(TicketRecord(
        id=tid, run_id=run_id, problem_id=pid, kb_article_id=None,
        ticket_type="l1", priority="low", status="resolved", subject="s",
        customer_persona_json="{}", agent_persona_json="{}",
        ground_truth_json=None, metadata_json=None,
        quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(timezone.utc), resolved_at=None,
    ))
    base = dict(
        ticket_id=tid, turn_index=0,
        speaker="customer", speaker_persona=None, content="x",
        intent=None, kb_references_json=None,
        noise_applied=False, noise_type=None, quality_flag=None,
        created_at=datetime.now(timezone.utc),
    )
    tu_repo.create(TurnRecord(id=str(uuid4()), **base))
    tu_repo.create(TurnRecord(id=str(uuid4()), **base))  # same turn_index — must be ignored
    assert len(tu_repo.list_for_ticket(tid)) == 1
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_repo_ticket_turn.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Append to `src/csfd/storage/repository.py`**

```python
@dataclass(slots=True)
class TicketRecord:
    id: str
    run_id: str
    problem_id: str
    kb_article_id: str | None
    ticket_type: str
    priority: str
    status: str
    subject: str
    customer_persona_json: str
    agent_persona_json: str
    ground_truth_json: str | None
    metadata_json: str | None
    quality_flag: str | None
    unresolved_issues_json: str | None
    created_at: datetime
    resolved_at: datetime | None


class TicketRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, t: TicketRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO tickets
                  (id, run_id, problem_id, kb_article_id, ticket_type, priority,
                   status, subject, customer_persona_json, agent_persona_json,
                   ground_truth_json, metadata_json,
                   quality_flag, unresolved_issues_json, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t.id, t.run_id, t.problem_id, t.kb_article_id, t.ticket_type,
                    t.priority, t.status, t.subject,
                    t.customer_persona_json, t.agent_persona_json,
                    t.ground_truth_json, t.metadata_json,
                    t.quality_flag, t.unresolved_issues_json,
                    t.created_at.isoformat(),
                    t.resolved_at.isoformat() if t.resolved_at else None,
                ),
            )


@dataclass(slots=True)
class TurnRecord:
    id: str
    ticket_id: str
    turn_index: int
    speaker: str
    speaker_persona: str | None
    content: str
    intent: str | None
    kb_references_json: str | None
    noise_applied: bool
    noise_type: str | None
    quality_flag: str | None
    created_at: datetime


class TurnRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, t: TurnRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO turns
                  (id, ticket_id, turn_index, speaker, speaker_persona, content,
                   intent, kb_references_json, noise_applied, noise_type,
                   quality_flag, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t.id, t.ticket_id, t.turn_index, t.speaker, t.speaker_persona,
                    t.content, t.intent, t.kb_references_json,
                    1 if t.noise_applied else 0, t.noise_type,
                    t.quality_flag, t.created_at.isoformat(),
                ),
            )

    def list_for_ticket(self, ticket_id: str) -> list[TurnRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM turns WHERE ticket_id = ? ORDER BY turn_index",
                (ticket_id,),
            ).fetchall()
        return [
            TurnRecord(
                id=r["id"], ticket_id=r["ticket_id"], turn_index=r["turn_index"],
                speaker=r["speaker"], speaker_persona=r["speaker_persona"],
                content=r["content"], intent=r["intent"],
                kb_references_json=r["kb_references_json"],
                noise_applied=bool(r["noise_applied"]),
                noise_type=r["noise_type"],
                quality_flag=r["quality_flag"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_repo_ticket_turn.py -v
```
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/repository.py tests/unit/test_repo_ticket_turn.py
git commit -m "feat: add TicketRepo and TurnRepo with idempotent inserts"
```

---

## Task 17: Repository — `AgentTraceRepo`

**Files:**
- Modify: `src/csfd/storage/repository.py`
- Test: `tests/unit/test_repo_trace.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_repo_trace.py`:
```python
from datetime import datetime, timezone
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    AgentTraceRecord, AgentTraceRepo, RunRecord, RunRepo,
)


def test_create_and_filter_traces(tmp_db_path):
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="running",
        started_at=datetime.now(timezone.utc), completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None, config_snapshot_json="{}",
        stats_json=None, error_summary=None,
    ))
    repo = AgentTraceRepo(db)
    for role in ("generator", "consistency", "background", "scenario"):
        repo.create(AgentTraceRecord(
            id=str(uuid4()), run_id=run_id, thread_id=run_id,
            node_name=f"{role}_node", agent_role=role,
            artifact_type="problem", artifact_id=None, attempt=0,
            prompt_id="abc123", input_json="{}", output_json=None,
            verdict=None, verdict_issues_json=None,
            model_provider="fake", model_id="fake-v1",
            tokens_in=10, tokens_out=20, cost_usd_estimated=0.001,
            latency_ms=100, parent_trace_id=None,
            status="ok", error_class=None, error_message=None,
            created_at=datetime.now(timezone.utc),
        ))
    rows = repo.list_for_run(run_id)
    assert len(rows) == 4
    only_gen = repo.list_for_run(run_id, agent_role="generator")
    assert len(only_gen) == 1
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_repo_trace.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Append to `src/csfd/storage/repository.py`**

```python
@dataclass(slots=True)
class AgentTraceRecord:
    id: str
    run_id: str
    thread_id: str
    node_name: str
    agent_role: str
    artifact_type: str
    artifact_id: str | None
    attempt: int
    prompt_id: str
    input_json: str
    output_json: str | None
    verdict: str | None
    verdict_issues_json: str | None
    model_provider: str
    model_id: str
    tokens_in: int | None
    tokens_out: int | None
    cost_usd_estimated: float | None
    latency_ms: int | None
    parent_trace_id: str | None
    status: str
    error_class: str | None
    error_message: str | None
    created_at: datetime


class AgentTraceRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, t: AgentTraceRecord) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_traces
                  (id, run_id, thread_id, node_name, agent_role,
                   artifact_type, artifact_id, attempt, prompt_id,
                   input_json, output_json, verdict, verdict_issues_json,
                   model_provider, model_id, tokens_in, tokens_out,
                   cost_usd_estimated, latency_ms, parent_trace_id,
                   status, error_class, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t.id, t.run_id, t.thread_id, t.node_name, t.agent_role,
                    t.artifact_type, t.artifact_id, t.attempt, t.prompt_id,
                    t.input_json, t.output_json, t.verdict, t.verdict_issues_json,
                    t.model_provider, t.model_id, t.tokens_in, t.tokens_out,
                    t.cost_usd_estimated, t.latency_ms, t.parent_trace_id,
                    t.status, t.error_class, t.error_message,
                    t.created_at.isoformat(),
                ),
            )

    def list_for_run(self, run_id: str, *, agent_role: str | None = None) -> list[AgentTraceRecord]:
        sql = "SELECT * FROM agent_traces WHERE run_id = ?"
        params: list[Any] = [run_id]
        if agent_role:
            sql += " AND agent_role = ?"
            params.append(agent_role)
        sql += " ORDER BY created_at"
        with self.db.connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            AgentTraceRecord(
                id=r["id"], run_id=r["run_id"], thread_id=r["thread_id"],
                node_name=r["node_name"], agent_role=r["agent_role"],
                artifact_type=r["artifact_type"], artifact_id=r["artifact_id"],
                attempt=r["attempt"], prompt_id=r["prompt_id"],
                input_json=r["input_json"], output_json=r["output_json"],
                verdict=r["verdict"], verdict_issues_json=r["verdict_issues_json"],
                model_provider=r["model_provider"], model_id=r["model_id"],
                tokens_in=r["tokens_in"], tokens_out=r["tokens_out"],
                cost_usd_estimated=r["cost_usd_estimated"],
                latency_ms=r["latency_ms"], parent_trace_id=r["parent_trace_id"],
                status=r["status"], error_class=r["error_class"],
                error_message=r["error_message"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_repo_trace.py -v
```
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/repository.py tests/unit/test_repo_trace.py
git commit -m "feat: add AgentTraceRepo with run + role filtering"
```

---

## Task 18: Exporters — JSONL

**Files:**
- Create: `src/csfd/storage/exporters.py`
- Test: `tests/unit/test_exporters_jsonl.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_exporters_jsonl.py`:
```python
import json
from datetime import datetime, timezone
from uuid import uuid4

from csfd.storage.db import Database
from csfd.storage.exporters import export_run_to_jsonl
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemRecord, ProblemRepo, RunRecord, RunRepo,
)


def test_export_jsonl_writes_problems_file(tmp_path):
    db_path = tmp_path / "x.sqlite"
    db = Database(path=db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="completed",
        started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
        run_seed=1, pipeline_version="0.1.0", git_sha=None,
        config_snapshot_json="{}", stats_json=None, error_summary=None,
    ))
    repo = ProblemRepo(db)
    for i in range(2):
        repo.create(ProblemRecord(
            id=str(uuid4()), run_id=run_id, title=f"P{i}",
            description="d", category="c", severity="low",
            has_kb=False, coverage_reasoning=None, coverage_confidence=None,
            metadata_json=None, quality_flag=None,
            unresolved_issues_json=None,
            created_at=datetime.now(timezone.utc),
        ))
    out = tmp_path / "exports"
    paths = export_run_to_jsonl(db, run_id, out_dir=out)
    problems_path = out / run_id / "problems.jsonl"
    assert problems_path in paths
    lines = problems_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["run_id"] == run_id
    manifest = json.loads((out / run_id / "manifest.json").read_text())
    assert "problems.jsonl" in manifest["files"]
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_exporters_jsonl.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/storage/exporters.py`**

```python
"""Export a run's artifacts to JSONL and Parquet."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from csfd.storage.db import Database
from csfd.storage.repository import (
    AgentTraceRepo, KBArticleRepo, ProblemRepo,
    TicketRepo, TurnRepo,
)


_TABLES = {
    "problems":     "SELECT * FROM problems WHERE run_id = ?",
    "kb_articles":  "SELECT * FROM kb_articles WHERE run_id = ?",
    "tickets":      "SELECT * FROM tickets WHERE run_id = ?",
    "turns":        ("SELECT turns.* FROM turns "
                     "JOIN tickets ON tickets.id = turns.ticket_id "
                     "WHERE tickets.run_id = ?"),
    "agent_traces": "SELECT * FROM agent_traces WHERE run_id = ?",
}


def _serialise(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def export_run_to_jsonl(db: Database, run_id: str, *, out_dir: Path) -> list[Path]:
    """Write one .jsonl per artifact table plus a manifest.json. Returns the list of files."""
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    file_checksums: dict[str, str] = {}

    with db.connect() as conn:
        for table, sql in _TABLES.items():
            path = run_dir / f"{table}.jsonl"
            rows = conn.execute(sql, (run_id,)).fetchall()
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    payload = {k: _serialise(r[k]) for k in r.keys()}
                    f.write(json.dumps(payload, separators=(",", ":"), default=str))
                    f.write("\n")
            written.append(path)
            file_checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "files": file_checksums,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return written
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_exporters_jsonl.py -v
```
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/exporters.py tests/unit/test_exporters_jsonl.py
git commit -m "feat: add JSONL exporter with per-run manifest"
```

---

## Task 19: Exporters — Parquet

**Files:**
- Modify: `src/csfd/storage/exporters.py`
- Test: `tests/unit/test_exporters_parquet.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_exporters_parquet.py`:
```python
from datetime import datetime, timezone
from uuid import uuid4

import pyarrow.parquet as pq

from csfd.storage.db import Database
from csfd.storage.exporters import export_run_to_parquet
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import ProblemRecord, ProblemRepo, RunRecord, RunRepo


def test_export_parquet_round_trip(tmp_path):
    db = Database(path=tmp_path / "x.sqlite")
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="completed",
        started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
        run_seed=1, pipeline_version="0.1.0", git_sha=None,
        config_snapshot_json="{}", stats_json=None, error_summary=None,
    ))
    ProblemRepo(db).create(ProblemRecord(
        id=str(uuid4()), run_id=run_id, title="P1",
        description="d", category="c", severity="low",
        has_kb=True, coverage_reasoning=None, coverage_confidence=None,
        metadata_json=None, quality_flag=None,
        unresolved_issues_json=None,
        created_at=datetime.now(timezone.utc),
    ))
    out = tmp_path / "exports"
    paths = export_run_to_parquet(db, run_id, out_dir=out)
    problems_path = out / run_id / "problems.parquet"
    assert problems_path in paths
    table = pq.read_table(problems_path)
    assert table.num_rows == 1
    assert "title" in table.column_names
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_exporters_parquet.py -v
```
Expected: `ImportError`.

- [ ] **Step 3: Append to `src/csfd/storage/exporters.py`**

```python
import pyarrow as pa
import pyarrow.parquet as pq


def export_run_to_parquet(db: Database, run_id: str, *, out_dir: Path) -> list[Path]:
    """Write one .parquet per artifact table. Returns the list of files."""
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with db.connect() as conn:
        for table, sql in _TABLES.items():
            rows = conn.execute(sql, (run_id,)).fetchall()
            if not rows:
                continue
            cols = list(rows[0].keys())
            data: dict[str, list[Any]] = {c: [] for c in cols}
            for r in rows:
                for c in cols:
                    data[c].append(_serialise(r[c]))
            path = run_dir / f"{table}.parquet"
            pq.write_table(pa.Table.from_pydict(data), path)
            written.append(path)
    return written
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_exporters_parquet.py -v
```
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/storage/exporters.py tests/unit/test_exporters_parquet.py
git commit -m "feat: add Parquet exporter per artifact table"
```

---

## Task 20: Seed parsers — `seeds/company.py`

**Files:**
- Create: `src/csfd/seeds/company.py`
- Create: `tests/fixtures/tiny_company_seed.md`
- Test: `tests/unit/test_seeds_company.py`

- [ ] **Step 1: Create the fixture** `tests/fixtures/tiny_company_seed.md`

```markdown
# Acme Cloud

## Products
- **AcmeDB** — managed database service
- **AcmeAuth** — identity platform

## Customer segments
- SMB
- Enterprise

## Tone of voice
Professional, concise, empathetic. Avoid emoji.

## KB style conventions
- Lead each article with a one-sentence problem statement.
- Use numbered steps for troubleshooting.
- Reference product names in **bold**.

## Policies
- Refunds within 30 days for SMB customers.
- Enterprise tickets require named-account escalation.
```

- [ ] **Step 2: Write the failing test**

`tests/unit/test_seeds_company.py`:
```python
from pathlib import Path

from csfd.seeds.company import CompanyProfile, parse_company_seed


def test_parse_company_seed_extracts_name_and_sections():
    profile: CompanyProfile = parse_company_seed(Path("tests/fixtures/tiny_company_seed.md"))
    assert profile.name == "Acme Cloud"
    assert "AcmeDB" in profile.raw_markdown
    assert "Products" in profile.sections
    assert "Tone of voice" in profile.sections


def test_parse_company_seed_preserves_source_path():
    profile = parse_company_seed(Path("tests/fixtures/tiny_company_seed.md"))
    assert profile.source_path.name == "tiny_company_seed.md"
```

- [ ] **Step 3: Run and verify it fails**

```bash
uv run pytest tests/unit/test_seeds_company.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/csfd/seeds/company.py`**

```python
"""Parse company_seed.md into a typed CompanyProfile."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(slots=True)
class CompanyProfile:
    name: str
    raw_markdown: str
    sections: dict[str, str] = field(default_factory=dict)
    source_path: Path = Path(".")


def parse_company_seed(path: Path) -> CompanyProfile:
    text = path.read_text(encoding="utf-8")
    h1 = _H1.search(text)
    name = h1.group(1).strip() if h1 else path.stem
    sections: dict[str, str] = {}
    headings = list(_H2.finditer(text))
    for i, m in enumerate(headings):
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections[m.group(1).strip()] = text[start:end].strip()
    return CompanyProfile(
        name=name,
        raw_markdown=text,
        sections=sections,
        source_path=path,
    )
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/unit/test_seeds_company.py -v
git add src/csfd/seeds/company.py tests/fixtures/tiny_company_seed.md tests/unit/test_seeds_company.py
git commit -m "feat: add CompanyProfile parser for company_seed.md"
```

---

## Task 21: Seed parsers — `seeds/scenarios.py`

**Files:**
- Create: `src/csfd/seeds/scenarios.py`
- Create: `tests/fixtures/tiny_scenarios_seed.md`
- Test: `tests/unit/test_seeds_scenarios.py`

- [ ] **Step 1: Create fixture** `tests/fixtures/tiny_scenarios_seed.md`

```markdown
# Scenarios

## Billing — overage dispute
Customer disputes a charge for usage beyond their plan.

## Billing — refund request
Customer requests a refund within the eligibility window.

## Authentication — locked account
Customer cannot log in; account is locked after failed attempts.

## Database — slow query
Customer reports query performance regression after a release.
```

- [ ] **Step 2: Write the failing test**

`tests/unit/test_seeds_scenarios.py`:
```python
from pathlib import Path

from csfd.seeds.scenarios import Scenario, ScenarioCatalogue, parse_scenarios_seed


def test_parse_scenarios_extracts_categories():
    cat: ScenarioCatalogue = parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md"))
    assert len(cat.scenarios) == 4
    cats = {s.category for s in cat.scenarios}
    assert {"Billing", "Authentication", "Database"} <= cats


def test_scenario_has_summary():
    cat = parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md"))
    s = next(x for x in cat.scenarios if "overage" in x.title.lower())
    assert s.category == "Billing"
    assert "overage" in s.title.lower()
    assert s.summary
```

- [ ] **Step 3: Run and verify it fails**

```bash
uv run pytest tests/unit/test_seeds_scenarios.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/csfd/seeds/scenarios.py`**

```python
"""Parse scenarios_seed.md into a typed ScenarioCatalogue."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_CATEGORY_SEP = re.compile(r"\s*[—–-]\s*")


@dataclass(slots=True)
class Scenario:
    category: str
    title: str           # the part after the en/em dash
    summary: str         # body text under the heading


@dataclass(slots=True)
class ScenarioCatalogue:
    scenarios: list[Scenario] = field(default_factory=list)
    source_path: Path = Path(".")


def parse_scenarios_seed(path: Path) -> ScenarioCatalogue:
    text = path.read_text(encoding="utf-8")
    headings = list(_H2.finditer(text))
    items: list[Scenario] = []
    for i, m in enumerate(headings):
        heading = m.group(1).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        parts = _CATEGORY_SEP.split(heading, maxsplit=1)
        if len(parts) == 2:
            category, title = parts[0].strip(), parts[1].strip()
        else:
            category, title = heading, heading
        items.append(Scenario(category=category, title=title, summary=body))
    return ScenarioCatalogue(scenarios=items, source_path=path)
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/unit/test_seeds_scenarios.py -v
git add src/csfd/seeds/scenarios.py tests/fixtures/tiny_scenarios_seed.md tests/unit/test_seeds_scenarios.py
git commit -m "feat: add ScenarioCatalogue parser for scenarios_seed.md"
```

---

## Task 22: `ticket_types/definitions.py`

**Files:**
- Create: `src/csfd/ticket_types/definitions.py`
- Test: `tests/unit/test_ticket_types.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_ticket_types.py`:
```python
from csfd.ticket_types.definitions import TicketType, TICKET_TYPE_METADATA


def test_ticket_type_enum_values():
    assert TicketType.DOCS_REQUEST.value == "docs_request"
    assert TicketType.L1.value == "l1"
    assert TicketType.L2.value == "l2"
    assert TicketType.L3.value == "l3"


def test_all_types_have_metadata():
    for t in TicketType:
        meta = TICKET_TYPE_METADATA[t]
        assert meta.avg_turns >= 1
        assert meta.persona_label
        assert meta.requires_kb in (True, False)


def test_l3_does_not_require_kb():
    assert TICKET_TYPE_METADATA[TicketType.L3].requires_kb is False


def test_docs_l1_l2_require_kb():
    for t in (TicketType.DOCS_REQUEST, TicketType.L1, TicketType.L2):
        assert TICKET_TYPE_METADATA[t].requires_kb is True
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_ticket_types.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/ticket_types/definitions.py`**

```python
"""Ticket type enum and per-type metadata."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TicketType(StrEnum):
    DOCS_REQUEST = "docs_request"
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"


@dataclass(slots=True, frozen=True)
class TicketTypeMetadata:
    avg_turns: int
    persona_label: str
    requires_kb: bool
    description: str


TICKET_TYPE_METADATA: dict[TicketType, TicketTypeMetadata] = {
    TicketType.DOCS_REQUEST: TicketTypeMetadata(
        avg_turns=2,
        persona_label="Documentation Rep",
        requires_kb=True,
        description="Customer asks for documentation; agent provides links + summary.",
    ),
    TicketType.L1: TicketTypeMetadata(
        avg_turns=3,
        persona_label="L1 Support",
        requires_kb=True,
        description="First-line rep; quick resolution; high reliance on KB.",
    ),
    TicketType.L2: TicketTypeMetadata(
        avg_turns=5,
        persona_label="L2 Support",
        requires_kb=True,
        description="Escalated; structured troubleshooting; KB + judgment.",
    ),
    TicketType.L3: TicketTypeMetadata(
        avg_turns=7,
        persona_label="L3 Domain Specialist",
        requires_kb=False,
        description="Domain specialist; novel/edge-case resolution; no KB available.",
    ),
}
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/unit/test_ticket_types.py -v
git add src/csfd/ticket_types/definitions.py tests/unit/test_ticket_types.py
git commit -m "feat: add TicketType enum with per-type metadata"
```

---

## Task 23: Prompt registry

**Files:**
- Create: `src/csfd/prompts/registry.py`
- Create: `prompts/phase1/_test.md.j2` (fixture for the test)
- Create: `prompts/phase2/_test.md.j2`
- Test: `tests/unit/test_prompt_registry.py`

- [ ] **Step 1: Create fixture prompt templates**

`prompts/phase1/_test.md.j2`:
```
Phase1 test template for {{ name }}.
```

`prompts/phase2/_test.md.j2`:
```
Phase2 test template for {{ name }}.
```

- [ ] **Step 2: Write the failing test**

`tests/unit/test_prompt_registry.py`:
```python
from pathlib import Path

import pytest

from csfd.prompts.registry import PromptRegistry


def test_registry_discovers_and_hashes_templates():
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    handle = reg.get("phase1._test")
    assert handle.name == "phase1._test"
    assert len(handle.version) == 12
    rendered = handle.template.render(name="world")
    assert "world" in rendered


def test_registry_hashes_are_stable_across_loads():
    a = PromptRegistry(root=Path("prompts"))
    a.load()
    b = PromptRegistry(root=Path("prompts"))
    b.load()
    assert a.get("phase1._test").version == b.get("phase1._test").version


def test_registry_unknown_name_raises():
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    with pytest.raises(KeyError):
        reg.get("nonexistent.template")
```

- [ ] **Step 3: Run and verify it fails**

```bash
uv run pytest tests/unit/test_prompt_registry.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/csfd/prompts/registry.py`**

```python
"""Prompt template registry with content-hash versioning."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from csfd.utils.hashing import short_hash


@dataclass(slots=True, frozen=True)
class PromptHandle:
    name: str            # e.g., "phase1.problem_generator"
    template: jinja2.Template
    source: str
    version: str         # short content hash


@dataclass(slots=True)
class PromptRegistry:
    root: Path
    _handles: dict[str, PromptHandle] = field(default_factory=dict)
    _env: jinja2.Environment | None = None

    def load(self) -> None:
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.root),
            keep_trailing_newline=True,
            autoescape=False,
        )
        self._handles.clear()
        for path in self.root.rglob("*.md.j2"):
            rel = path.relative_to(self.root).with_suffix("").with_suffix("")
            # rel = phase1/problem_generator (after stripping .md.j2)
            name = ".".join(rel.parts)
            source = path.read_text(encoding="utf-8")
            template = self._env.from_string(source)
            self._handles[name] = PromptHandle(
                name=name, template=template, source=source,
                version=short_hash(source),
            )

    def get(self, name: str) -> PromptHandle:
        if not self._handles:
            raise RuntimeError("Registry not loaded; call .load() first")
        if name not in self._handles:
            raise KeyError(name)
        return self._handles[name]

    def all(self) -> dict[str, PromptHandle]:
        return dict(self._handles)
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/unit/test_prompt_registry.py -v
git add src/csfd/prompts/registry.py prompts/phase1/_test.md.j2 prompts/phase2/_test.md.j2 tests/unit/test_prompt_registry.py
git commit -m "feat: add PromptRegistry with jinja2 + content-hash versioning"
```

---

## Task 24: LLM provider registry (with `FakeChatModel`)

**Files:**
- Create: `src/csfd/models/registry.py`
- Create: `src/csfd/models/fake.py` (for tests, but lives in src so prod code can import it during tests)
- Test: `tests/unit/test_models_registry.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models_registry.py`:
```python
import pytest

from csfd.models.registry import build_llm
from csfd.settings import AgentLLMConfig


def test_build_anthropic_llm_constructs_client():
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.1)
    llm = build_llm(cfg)
    # don't invoke (no API key); just check type
    assert type(llm).__name__ == "ChatAnthropic"


def test_build_openai_compat_llm_uses_base_url():
    cfg = AgentLLMConfig(
        provider="openai_compat", model="llama3.1:8b",
        temperature=0.1, base_url="http://localhost:11434/v1",
        api_key="local",
    )
    llm = build_llm(cfg)
    assert type(llm).__name__ == "ChatOpenAI"


def test_build_fake_llm_returns_canned_response():
    cfg = AgentLLMConfig(provider="anthropic", model="fake", temperature=0)
    from csfd.models.fake import FakeChatModel
    llm = FakeChatModel(canned={"default": "hello"})
    result = llm.invoke("anything")
    assert "hello" in result.content


def test_build_llm_unknown_provider_raises():
    cfg = AgentLLMConfig.model_construct(provider="nope", model="x", temperature=0)
    with pytest.raises(ValueError):
        build_llm(cfg)
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_models_registry.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/models/registry.py`**

```python
"""Build LangChain chat models from AgentLLMConfig."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from csfd.settings import AgentLLMConfig


def build_llm(cfg: AgentLLMConfig) -> BaseChatModel:
    if cfg.provider == "anthropic":
        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens or 4096,
            timeout=cfg.timeout_s,
        )
    if cfg.provider == "openai_compat":
        return ChatOpenAI(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key or "local",
            temperature=cfg.temperature,
            timeout=cfg.timeout_s,
        )
    raise ValueError(f"Unknown provider: {cfg.provider}")
```

- [ ] **Step 4: Implement `src/csfd/models/fake.py`**

```python
"""In-process fake chat model for unit and integration tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


@dataclass
class FakeChatModel(BaseChatModel):
    canned: dict[str, str] = field(default_factory=dict)
    call_log: list[Any] = field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_log.append(messages)
        content = self.canned.get("default", "")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/unit/test_models_registry.py -v
git add src/csfd/models/registry.py src/csfd/models/fake.py tests/unit/test_models_registry.py
git commit -m "feat: add LLM provider registry + FakeChatModel"
```

---

## Task 25: Budget tracker

**Files:**
- Create: `src/csfd/budget/tracker.py`
- Test: `tests/unit/test_budget_tracker.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_budget_tracker.py`:
```python
import pytest

from csfd.budget.tracker import BudgetTracker
from csfd.errors import BudgetExceededError


def test_tracker_accumulates_tokens_and_cost():
    t = BudgetTracker(max_tokens=1000, max_usd=1.0)
    t.record(tokens_in=100, tokens_out=50, cost_usd=0.01)
    t.record(tokens_in=200, tokens_out=100, cost_usd=0.02)
    stats = t.stats()
    assert stats["tokens"] == 450
    assert stats["usd"] == pytest.approx(0.03)


def test_tracker_raises_when_tokens_exceeded():
    t = BudgetTracker(max_tokens=300, max_usd=10.0)
    t.record(tokens_in=200, tokens_out=50, cost_usd=0.01)
    with pytest.raises(BudgetExceededError):
        t.record(tokens_in=100, tokens_out=0, cost_usd=0.01)


def test_tracker_raises_when_usd_exceeded():
    t = BudgetTracker(max_tokens=10_000, max_usd=0.05)
    t.record(tokens_in=10, tokens_out=10, cost_usd=0.03)
    with pytest.raises(BudgetExceededError):
        t.record(tokens_in=10, tokens_out=10, cost_usd=0.03)
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_budget_tracker.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/budget/tracker.py`**

```python
"""Per-run token + USD budget tracker."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from csfd.errors import BudgetExceededError


@dataclass(slots=True)
class BudgetTracker:
    max_tokens: int
    max_usd: float
    _tokens_used: int = field(default=0, init=False)
    _usd_used: float = field(default=0.0, init=False)

    def record(self, *, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        self._tokens_used += tokens_in + tokens_out
        self._usd_used += cost_usd
        self._check()

    def _check(self) -> None:
        if self._tokens_used > self.max_tokens or self._usd_used > self.max_usd:
            raise BudgetExceededError(stats=self.stats())

    def stats(self) -> dict[str, Any]:
        return {
            "tokens": self._tokens_used,
            "usd": round(self._usd_used, 6),
            "max_tokens": self.max_tokens,
            "max_usd": self.max_usd,
        }
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/unit/test_budget_tracker.py -v
git add src/csfd/budget/tracker.py tests/unit/test_budget_tracker.py
git commit -m "feat: add BudgetTracker enforcing per-run token + USD limits"
```

---

## Task 26: Circuit breaker

**Files:**
- Create: `src/csfd/budget/breaker.py`
- Test: `tests/unit/test_budget_breaker.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_budget_breaker.py`:
```python
import pytest
from pybreaker import CircuitBreakerError

from csfd.budget.breaker import build_breaker
from csfd.errors import TransportError


def test_breaker_trips_after_threshold():
    br = build_breaker(fail_max=3, reset_timeout=60)

    @br
    def always_fail() -> None:
        raise TransportError("boom")

    for _ in range(3):
        with pytest.raises(TransportError):
            always_fail()
    with pytest.raises(CircuitBreakerError):
        always_fail()


def test_breaker_resets_on_success():
    br = build_breaker(fail_max=2, reset_timeout=60)

    state = {"fail": True}

    @br
    def maybe_fail() -> str:
        if state["fail"]:
            raise TransportError("nope")
        return "ok"

    with pytest.raises(TransportError):
        maybe_fail()
    state["fail"] = False
    assert maybe_fail() == "ok"
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_budget_breaker.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/budget/breaker.py`**

```python
"""Circuit breaker around LLM transport. Trips on repeated TransportError."""
from __future__ import annotations

from pybreaker import CircuitBreaker

from csfd.errors import TransportError


def build_breaker(*, fail_max: int = 5, reset_timeout: int = 60) -> CircuitBreaker:
    """Return a CircuitBreaker that trips only on TransportError instances."""
    return CircuitBreaker(
        fail_max=fail_max,
        reset_timeout=reset_timeout,
        exclude=[lambda e: not isinstance(e, TransportError)],
    )
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/unit/test_budget_breaker.py -v
git add src/csfd/budget/breaker.py tests/unit/test_budget_breaker.py
git commit -m "feat: add pybreaker circuit breaker for transport errors"
```

---

## Task 27: LangSmith opt-in setup

**Files:**
- Create: `src/csfd/observability/langsmith.py`
- Test: `tests/unit/test_langsmith_optin.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_langsmith_optin.py`:
```python
import os

from csfd.observability.langsmith import maybe_enable_langsmith


def test_no_op_when_api_key_absent(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    maybe_enable_langsmith(api_key=None, project="csfd")
    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_sets_env_when_api_key_present(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    maybe_enable_langsmith(api_key="lsk_test", project="my-proj")
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "my-proj"
    assert os.environ["LANGCHAIN_API_KEY"] == "lsk_test"
```

- [ ] **Step 2: Run and verify it fails**

```bash
uv run pytest tests/unit/test_langsmith_optin.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/csfd/observability/langsmith.py`**

```python
"""Opt-in LangSmith tracing setup. No-op unless an API key is provided."""
from __future__ import annotations

import os


def maybe_enable_langsmith(*, api_key: str | None, project: str) -> bool:
    """Enable LangChain/LangSmith tracing if an API key is present.

    Returns True if tracing was enabled. False otherwise.
    """
    if not api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    return True
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/unit/test_langsmith_optin.py -v
git add src/csfd/observability/langsmith.py tests/unit/test_langsmith_optin.py
git commit -m "feat: add opt-in LangSmith tracing setup"
```

---

## Task 28: Full-suite verification and mypy gate

**Files:**
- Modify: `pyproject.toml` (no change expected; verify config)
- Add: `tests/unit/test_smoke_imports.py`

- [ ] **Step 1: Write a smoke import test**

`tests/unit/test_smoke_imports.py`:
```python
def test_all_top_level_modules_import():
    import csfd
    import csfd.errors
    import csfd.settings
    import csfd.utils.hashing
    import csfd.utils.rng
    import csfd.utils.retry
    import csfd.observability.logging
    import csfd.observability.langsmith
    import csfd.storage.db
    import csfd.storage.repository
    import csfd.storage.exporters
    import csfd.storage.migrations.runner
    import csfd.seeds.company
    import csfd.seeds.scenarios
    import csfd.ticket_types.definitions
    import csfd.prompts.registry
    import csfd.models.registry
    import csfd.models.fake
    import csfd.budget.tracker
    import csfd.budget.breaker
    assert csfd is not None
```

- [ ] **Step 2: Run the full test suite**

```bash
uv run pytest -q
```
Expected: every test passes (50+ tests across all task files).

- [ ] **Step 3: Run mypy strict on all source and tests**

```bash
uv run mypy src tests
```
Expected: `Success: no issues found in N source files`. If issues surface, fix them inline and re-run.

- [ ] **Step 4: Run ruff**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```
Expected: passes; format reports no changes needed.

- [ ] **Step 5: Install pre-commit hook and commit final state**

```bash
uv run pre-commit install
git add tests/unit/test_smoke_imports.py
git commit -m "test: add smoke imports + verify full suite under mypy and ruff"
```

---

## Plan completion criteria

- [ ] All 28 tasks complete; their tests pass individually.
- [ ] `uv run pytest -q` is green across the full suite.
- [ ] `uv run mypy --strict src tests` reports no issues.
- [ ] `uv run ruff check src tests` and `ruff format --check src tests` pass.
- [ ] `git log --oneline` shows ~28 conventional commits (`chore:` / `feat:` / `test:`).
- [ ] The repo can:
  - Migrate a fresh SQLite database via `apply_migrations(...)`.
  - Round-trip records through every `*Repo`.
  - Load `config/default.yaml` and merge profiles.
  - Parse the two tiny seed fixtures into typed objects.
  - Discover and content-hash prompt templates.
  - Build a `ChatAnthropic` / `ChatOpenAI` / `FakeChatModel` from agent config.
  - Accumulate token + USD budgets and raise on overflow.

After all boxes are checked, Plan 1 is complete. Hand off to **Plan 2 — Agent Framework**, which builds the `AgentRole` base, the `Verdict` / `Issue` schemas, and the concrete `Generator` / `Checker` / `CreativeNoise` classes on top of this foundation.
