# CSFD Phase 1 — KB Generation Implementation Plan (Plan 3 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Build the Phase 1 LangGraph subgraph that generates a problem pool and KB articles from `company_seed.md` + `scenarios_seed.md`. After Plan 3: a runnable `phase1` subgraph that, given a tiny seed pair and a `FakeChatModel`, produces N candidate problems → dedups → runs 3 parallel checkers via `Send` → coverage-decides each → writes KB articles for covered problems → another 3-checker pass → persists problems and articles to SQLite. End-to-end integration test exercises the full subgraph against fixtures with no LLM key.

**Architecture:**
- `KBState` is a Pydantic BaseModel with `Annotated[list[Verdict], operator.add]` reducers for fan-in.
- Nodes are async functions `(KBState) -> dict[str, Any]` returning partial state updates.
- Three checkers fan out via `Send` and aggregate via the verdict reducer; `Overwrite(value=[])` resets verdicts on retry.
- Coverage decider is a hybrid: an LLM-based `coverage_judge` Generator emits per-problem `CoverageDecision`, then a deterministic top-up/trim step adjusts to hit `kb_coverage_target_rate`.
- Article generation has its own check loop (separate from the problem check loop), reusing the same Checker pattern.
- Bounded retries via `max_retries_per_artifact`; on exhaustion: `commit_with_warning` (the artifact is persisted with `quality_flag` set).

**Tech Stack:** Builds on Plans 1+2. Adds: `langgraph` `StateGraph` / `Send` / `Overwrite` / conditional edges, lexical TF-IDF dedup (stdlib + simple cosine), async LangGraph nodes.

**Working directory:** `/Users/joao.augusto/Documents/Customer Service Fake Data`

---

## File Structure (Plan 3 scope)

| Path | Responsibility |
|---|---|
| `src/csfd/phases/phase1_kb/state.py` | `KBState`, `ProblemDraft`, `KBArticleDraft`, `CoverageDecision` Pydantic models |
| `src/csfd/phases/phase1_kb/nodes.py` | Async node functions (seed_load, problem_brainstorm, dedup, coverage_decider, article_writer, persisters) |
| `src/csfd/phases/phase1_kb/routing.py` | `Send`-based fan-out + verdict-aggregation routing logic |
| `src/csfd/phases/phase1_kb/subgraph.py` | `build_phase1_graph(...)` — composes the StateGraph |
| `src/csfd/phases/phase1_kb/dedup.py` | `lexical_dedup(problems, threshold)` — TF-IDF cosine over problem titles+descriptions |
| `src/csfd/phases/phase1_kb/__init__.py` | re-exports |
| `tests/unit/test_phase1_state.py` | KBState shape + reducer tests |
| `tests/unit/test_phase1_dedup.py` | lexical dedup unit tests |
| `tests/unit/test_phase1_nodes.py` | per-node tests (seed_load, brainstorm, etc.) |
| `tests/unit/test_phase1_routing.py` | Send + aggregation routing tests |
| `tests/integration/test_phase1_e2e_tiny.py` | Phase 1 subgraph end-to-end with FakeChatModel + tiny seeds |

Out of scope: real LLM integration tests (Plan 5 cassettes), Phase 2 anything, parent graph composition (Plan 5).

---

## Task 1: `KBState` + draft models

**Files:**
- Create: `src/csfd/phases/phase1_kb/state.py`
- Test: `tests/unit/test_phase1_state.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase1_state.py`:
```python
from datetime import datetime, timezone
from uuid import uuid4

from csfd.agents.base import Issue, Verdict
from csfd.phases.phase1_kb.state import (
    CoverageDecision,
    KBArticleDraft,
    KBState,
    PhaseStats,
    ProblemDraft,
)
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue


def _empty_company() -> CompanyProfile:
    return CompanyProfile(name="Acme", raw_markdown="# Acme")


def _empty_scenarios() -> ScenarioCatalogue:
    return ScenarioCatalogue()


def test_kb_state_defaults() -> None:
    state = KBState(
        run_id=str(uuid4()),
        run_seed=42,
        company=_empty_company(),
        scenarios=_empty_scenarios(),
    )
    assert state.problems_committed == []
    assert state.current_problem_draft is None
    assert state.current_article_draft is None
    assert state.verdicts == []
    assert state.retry_attempt == 0
    assert isinstance(state.stats, PhaseStats)


def test_kb_state_verdict_reducer_appends() -> None:
    """Verify the Annotated reducer for verdicts is operator.add (list concat)."""
    from typing import get_type_hints, get_args
    hints = get_type_hints(KBState, include_extras=True)
    annotated = hints["verdicts"]
    args = get_args(annotated)
    # First arg is the inner type, second is the reducer
    import operator
    assert args[1] is operator.add


def test_problem_draft_required_fields() -> None:
    p = ProblemDraft(
        title="Login fails",
        description="Customer cannot log in",
        category="Authentication",
        severity="medium",
    )
    assert p.title == "Login fails"


def test_kb_article_draft_required_fields() -> None:
    a = KBArticleDraft(
        title="How to reset",
        content_markdown="## Step 1\n...",
        troubleshooting_steps=[{"step": "do X", "expected_result": "Y"}],
    )
    assert a.title == "How to reset"
    assert len(a.troubleshooting_steps) == 1


def test_coverage_decision_required_fields() -> None:
    d = CoverageDecision(has_kb=True, reasoning="Common issue", confidence="high")
    assert d.has_kb is True
```

- [ ] **Step 2: Confirm failure** — `uv run pytest tests/unit/test_phase1_state.py -v` → ImportError.

- [ ] **Step 3: Implement `src/csfd/phases/phase1_kb/state.py`**

```python
"""KBState + draft Pydantic models for the Phase 1 KB-generation subgraph."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from csfd.agents.base import Verdict
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue


class ProblemDraft(BaseModel):
    """An LLM-generated candidate problem before persistence."""

    title: str
    description: str
    category: str
    severity: Literal["low", "medium", "high", "critical"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class KBArticleDraft(BaseModel):
    """An LLM-generated KB article before persistence."""

    title: str
    content_markdown: str
    troubleshooting_steps: list[dict[str, str]] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoverageDecision(BaseModel):
    """Output of the coverage_judge agent: does this problem warrant a KB article?"""

    has_kb: bool
    reasoning: str
    confidence: Literal["low", "medium", "high"]


class PhaseStats(BaseModel):
    problems_generated: int = 0
    problems_committed: int = 0
    articles_committed: int = 0
    total_tokens: int = 0
    total_usd: float = 0.0


class CommittedProblem(BaseModel):
    """A problem committed to the run state (paired with its CoverageDecision)."""

    id: str
    draft: ProblemDraft
    has_kb: bool
    coverage_reasoning: str
    coverage_confidence: str
    quality_flag: str | None = None
    unresolved_issues: list[Verdict] = Field(default_factory=list)


class CommittedArticle(BaseModel):
    """A KB article committed (paired with its problem id)."""

    id: str
    problem_id: str
    draft: KBArticleDraft
    quality_flag: str | None = None
    unresolved_issues: list[Verdict] = Field(default_factory=list)


class KBState(BaseModel):
    """In-memory working state for the Phase 1 LangGraph subgraph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    run_seed: int
    company: CompanyProfile
    scenarios: ScenarioCatalogue

    problems_committed: list[CommittedProblem] = Field(default_factory=list)
    articles_committed: list[CommittedArticle] = Field(default_factory=list)

    current_problem_draft: ProblemDraft | None = None
    current_article_draft: KBArticleDraft | None = None

    verdicts: Annotated[list[Verdict], operator.add] = Field(default_factory=list)
    retry_attempt: int = 0

    stats: PhaseStats = Field(default_factory=PhaseStats)
```

- [ ] **Step 4: Verify** — pytest 5 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase1_kb/state.py tests/unit/test_phase1_state.py
git commit -m "feat: add KBState and Phase 1 draft models"
```

---

## Task 2: Lexical TF-IDF dedup

**Files:**
- Create: `src/csfd/phases/phase1_kb/dedup.py`
- Test: `tests/unit/test_phase1_dedup.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase1_dedup.py`:
```python
from csfd.phases.phase1_kb.dedup import lexical_dedup
from csfd.phases.phase1_kb.state import ProblemDraft


def _p(title: str, desc: str = "") -> ProblemDraft:
    return ProblemDraft(title=title, description=desc, category="x", severity="low")


def test_lexical_dedup_removes_near_duplicates() -> None:
    items = [
        _p("Cannot log in to my account", "I get a 401 error"),
        _p("Cannot log into my account", "I get a 401 error"),  # near dup
        _p("Billing overage charge", "Charged for usage above plan"),
    ]
    keep = lexical_dedup(items, threshold=0.85)
    titles = sorted(p.title for p in keep)
    assert "Billing overage charge" in titles
    # Only one of the two near-duplicate logins survives
    login_count = sum(1 for p in keep if "log" in p.title.lower())
    assert login_count == 1


def test_lexical_dedup_keeps_distinct_items() -> None:
    items = [
        _p("Issue A", "first"),
        _p("Issue B", "second"),
        _p("Issue C", "third"),
    ]
    keep = lexical_dedup(items, threshold=0.85)
    assert len(keep) == 3


def test_lexical_dedup_empty_input() -> None:
    assert lexical_dedup([], threshold=0.85) == []
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Implement `src/csfd/phases/phase1_kb/dedup.py`**

```python
"""Stdlib-only lexical (TF-IDF cosine) dedup for ProblemDraft items."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from csfd.phases.phase1_kb.state import ProblemDraft

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _tf(tokens: list[str]) -> dict[str, float]:
    n = len(tokens) or 1
    counts = Counter(tokens)
    return {t: c / n for t, c in counts.items()}


def _idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    n_docs = len(corpus_tokens) or 1
    df: Counter[str] = Counter()
    for toks in corpus_tokens:
        for t in set(toks):
            df[t] += 1
    return {t: math.log((1 + n_docs) / (1 + df_t)) + 1.0 for t, df_t in df.items()}


def _vector(tf_doc: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    return {t: tf_v * idf.get(t, 0.0) for t, tf_v in tf_doc.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a if t in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def lexical_dedup(
    problems: Iterable[ProblemDraft],
    *,
    threshold: float,
) -> list[ProblemDraft]:
    """Return a deduplicated list — drop problems with cosine similarity above
    `threshold` to any earlier-kept problem."""
    items = list(problems)
    if not items:
        return []
    docs = [_tokenize(f"{p.title} {p.description}") for p in items]
    idf = _idf(docs)
    vectors = [_vector(_tf(d), idf) for d in docs]

    keep_indices: list[int] = []
    for i, vi in enumerate(vectors):
        is_dup = False
        for j in keep_indices:
            if _cosine(vi, vectors[j]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep_indices.append(i)
    return [items[i] for i in keep_indices]
```

- [ ] **Step 4: Verify** — pytest 3 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase1_kb/dedup.py tests/unit/test_phase1_dedup.py
git commit -m "feat: add lexical TF-IDF dedup for ProblemDraft items"
```

---

## Task 3: `seed_load` + `problem_brainstorm` nodes

**Files:**
- Create: `src/csfd/phases/phase1_kb/nodes.py` (start)
- Test: `tests/unit/test_phase1_nodes_seed_brainstorm.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase1_nodes_seed_brainstorm.py`:
```python
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import problem_brainstorm_node
from csfd.phases.phase1_kb.state import KBState, ProblemDraft
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import parse_company_seed
from csfd.seeds.scenarios import parse_scenarios_seed
from csfd.settings import AgentLLMConfig


def _state() -> KBState:
    return KBState(
        run_id=str(uuid4()),
        run_seed=1,
        company=parse_company_seed(Path("tests/fixtures/tiny_company_seed.md")),
        scenarios=parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md")),
    )


def _factory(canned_problem: ProblemDraft) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={ProblemDraft: canned_problem}),
        agent_configs={
            "problem_brainstorm": AgentLLMConfig(
                provider="anthropic", model="claude-haiku-4-5", temperature=0.5,
            ),
        },
    )


@pytest.mark.asyncio
async def test_problem_brainstorm_emits_draft_into_state() -> None:
    canned = ProblemDraft(
        title="Cannot reset password",
        description="Customer locked out after failed attempts",
        category="Authentication",
        severity="medium",
    )
    factory = _factory(canned)
    state = _state()
    update = await problem_brainstorm_node(state, factory=factory)
    assert "current_problem_draft" in update
    draft = update["current_problem_draft"]
    assert isinstance(draft, ProblemDraft)
    assert draft.title == "Cannot reset password"
```

- [ ] **Step 2: Confirm failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `src/csfd/phases/phase1_kb/nodes.py`** (initial — only seed_load + problem_brainstorm; later tasks append more nodes)

```python
"""Async LangGraph nodes for the Phase 1 KB-generation subgraph."""
from __future__ import annotations

from typing import Any

from csfd.agents.base import AgentContext
from csfd.agents.factory import AgentFactory
from csfd.phases.phase1_kb.state import KBState, ProblemDraft


async def seed_load_node(state: KBState) -> dict[str, Any]:
    """No-op node: the seeds are already loaded into KBState at run construction.

    Kept as an explicit graph entrypoint for readability and so future variations
    (e.g. enrichment) have a stable home.
    """
    return {}


async def problem_brainstorm_node(
    state: KBState,
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    """Generate a candidate Problem via the `problem_brainstorm` Generator agent."""
    gen = factory.build_generator(
        name="problem_brainstorm",
        prompt_name="phase1.problem_generator",
        output_schema_factory=lambda: ProblemDraft,
    )
    ctx = AgentContext(
        inputs={
            "company_name": state.company.name,
            "company_sections": state.company.sections,
            "scenarios": [
                {"category": s.category, "title": s.title, "summary": s.summary}
                for s in state.scenarios.scenarios
            ],
        },
        prior_verdicts=state.verdicts,
        retry_attempt=state.retry_attempt,
    )
    draft = await gen.invoke(ctx)
    if not isinstance(draft, ProblemDraft):
        raise TypeError(f"Expected ProblemDraft, got {type(draft).__name__}")
    return {"current_problem_draft": draft}
```

- [ ] **Step 4: Verify** — pytest 1 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase1_kb/nodes.py tests/unit/test_phase1_nodes_seed_brainstorm.py
git commit -m "feat: add seed_load + problem_brainstorm nodes for Phase 1"
```

---

## Task 4: `dedup_node` + `coverage_decider_node`

**Files:**
- Modify (append): `src/csfd/phases/phase1_kb/nodes.py`
- Test: `tests/unit/test_phase1_nodes_dedup_coverage.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase1_nodes_dedup_coverage.py`:
```python
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import coverage_decider_node, dedup_node
from csfd.phases.phase1_kb.state import (
    CommittedProblem,
    CoverageDecision,
    KBState,
    ProblemDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.settings import AgentLLMConfig


def _state(problems: list[CommittedProblem]) -> KBState:
    return KBState(
        run_id=str(uuid4()),
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        scenarios=ScenarioCatalogue(),
        problems_committed=problems,
    )


def _factory(canned: CoverageDecision) -> AgentFactory:
    (Path("prompts") / "phase1").mkdir(exist_ok=True)
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={CoverageDecision: canned}),
        agent_configs={
            "coverage_judge": AgentLLMConfig(
                provider="anthropic", model="claude-haiku-4-5", temperature=0.2,
            ),
        },
    )


def test_dedup_node_removes_near_duplicates() -> None:
    pcs = [
        CommittedProblem(
            id="1",
            draft=ProblemDraft(title=f"Login fails {i}", description="401 error",
                               category="auth", severity="low"),
            has_kb=False, coverage_reasoning="", coverage_confidence="low",
        )
        for i in range(2)
    ] + [
        CommittedProblem(
            id="3",
            draft=ProblemDraft(title="Billing overage", description="Charged more",
                               category="billing", severity="medium"),
            has_kb=False, coverage_reasoning="", coverage_confidence="low",
        ),
    ]
    state = _state(pcs)
    update = dedup_node(state, similarity_threshold=0.85)
    assert "problems_committed" in update
    deduped = update["problems_committed"]
    assert len(deduped) == 2
    titles = sorted(p.draft.title for p in deduped)
    assert "Billing overage" in titles


@pytest.mark.asyncio
async def test_coverage_decider_uses_judge_then_balances_to_target() -> None:
    """coverage_decider judges each problem then top-up/trims toward kb_coverage_target_rate."""
    pcs = [
        CommittedProblem(
            id=str(i),
            draft=ProblemDraft(title=f"P{i}", description=f"d{i}",
                               category="c", severity="low"),
            has_kb=False, coverage_reasoning="", coverage_confidence="low",
        )
        for i in range(10)
    ]
    state = _state(pcs)
    canned = CoverageDecision(has_kb=True, reasoning="ok", confidence="medium")
    factory = _factory(canned)
    update = await coverage_decider_node(
        state, factory=factory, target_rate=0.7,
    )
    out = update["problems_committed"]
    has_kb_count = sum(1 for p in out if p.has_kb)
    # 70% of 10 = 7
    assert has_kb_count == 7
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Append to `src/csfd/phases/phase1_kb/nodes.py`**

```python
from csfd.phases.phase1_kb.dedup import lexical_dedup
from csfd.phases.phase1_kb.state import (
    CommittedProblem,
    CoverageDecision,
)


def dedup_node(state: KBState, *, similarity_threshold: float) -> dict[str, Any]:
    """Drop near-duplicate committed problems via lexical TF-IDF cosine."""
    drafts = [p.draft for p in state.problems_committed]
    kept_drafts = lexical_dedup(drafts, threshold=similarity_threshold)
    kept_ids = {id(d) for d in kept_drafts}
    kept = [p for p in state.problems_committed if id(p.draft) in kept_ids]
    return {"problems_committed": kept}


async def coverage_decider_node(
    state: KBState,
    *,
    factory: AgentFactory,
    target_rate: float,
) -> dict[str, Any]:
    """For each committed problem: invoke the coverage_judge to decide has_kb,
    then top-up or trim borderline cases to hit `target_rate`."""
    judge = factory.build_generator(
        name="coverage_judge",
        prompt_name="phase1.problem_generator",  # placeholder; real prompt in Plan 5
        output_schema_factory=lambda: CoverageDecision,
    )
    decided: list[tuple[CommittedProblem, CoverageDecision]] = []
    for p in state.problems_committed:
        ctx = AgentContext(inputs={
            "problem": p.draft.model_dump(),
            "company_name": state.company.name,
        })
        decision = await judge.invoke(ctx)
        if not isinstance(decision, CoverageDecision):
            raise TypeError(f"Expected CoverageDecision, got {type(decision).__name__}")
        decided.append((p, decision))

    # Apply target rate by flipping borderline (confidence=low) cases
    n = len(decided)
    target_yes = round(n * target_rate)
    current_yes = sum(1 for _, d in decided if d.has_kb)

    if current_yes != target_yes:
        diff = target_yes - current_yes  # positive = need more yes
        flippable = [
            (idx, d) for idx, (_, d) in enumerate(decided)
            if (diff > 0 and not d.has_kb) or (diff < 0 and d.has_kb)
        ]
        # Prefer low-confidence flips
        flippable.sort(key=lambda x: 0 if x[1].confidence == "low" else 1)
        for idx, _ in flippable[:abs(diff)]:
            p, d = decided[idx]
            decided[idx] = (
                p,
                CoverageDecision(
                    has_kb=not d.has_kb,
                    reasoning=f"Adjusted from '{d.reasoning}' to hit target rate",
                    confidence=d.confidence,
                ),
            )

    updated = [
        CommittedProblem(
            id=p.id,
            draft=p.draft,
            has_kb=d.has_kb,
            coverage_reasoning=d.reasoning,
            coverage_confidence=d.confidence,
            quality_flag=p.quality_flag,
            unresolved_issues=p.unresolved_issues,
        )
        for (p, d) in decided
    ]
    return {"problems_committed": updated}
```

- [ ] **Step 4: Verify** — pytest 2 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase1_kb/nodes.py tests/unit/test_phase1_nodes_dedup_coverage.py
git commit -m "feat: add dedup + coverage_decider nodes for Phase 1"
```

---

## Task 5: `article_writer_node`

**Files:**
- Modify (append): `src/csfd/phases/phase1_kb/nodes.py`
- Test: `tests/unit/test_phase1_nodes_article.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase1_nodes_article.py`:
```python
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import article_writer_node
from csfd.phases.phase1_kb.state import (
    CommittedProblem,
    KBArticleDraft,
    KBState,
    ProblemDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.settings import AgentLLMConfig


@pytest.mark.asyncio
async def test_article_writer_produces_draft_for_kb_problem() -> None:
    pc = CommittedProblem(
        id="abc",
        draft=ProblemDraft(title="Login fails", description="401 error",
                           category="auth", severity="medium"),
        has_kb=True, coverage_reasoning="ok", coverage_confidence="high",
    )
    state = KBState(
        run_id=str(uuid4()), run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        scenarios=ScenarioCatalogue(),
        problems_committed=[pc],
        current_problem_draft=pc.draft,  # writer reads current draft + problem id
    )
    canned_article = KBArticleDraft(
        title="How to recover login",
        content_markdown="## Step 1\n...",
        troubleshooting_steps=[{"step": "do X", "expected_result": "Y"}],
    )
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    factory = AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={KBArticleDraft: canned_article}),
        agent_configs={
            "article_writer": AgentLLMConfig(
                provider="anthropic", model="claude-haiku-4-5", temperature=0.3,
            ),
        },
    )
    update = await article_writer_node(state, factory=factory, problem=pc)
    assert "current_article_draft" in update
    draft = update["current_article_draft"]
    assert isinstance(draft, KBArticleDraft)
    assert draft.title == "How to recover login"
```

- [ ] **Step 2: Confirm failure** — ImportError on `article_writer_node`.

- [ ] **Step 3: Append to `src/csfd/phases/phase1_kb/nodes.py`**

```python
from csfd.phases.phase1_kb.state import KBArticleDraft


async def article_writer_node(
    state: KBState,
    *,
    factory: AgentFactory,
    problem: CommittedProblem,
) -> dict[str, Any]:
    """Write a KB article for the given covered problem."""
    writer = factory.build_generator(
        name="article_writer",
        prompt_name="phase1.problem_generator",  # placeholder; real prompt in Plan 5
        output_schema_factory=lambda: KBArticleDraft,
    )
    ctx = AgentContext(
        inputs={
            "problem": problem.draft.model_dump(),
            "company_name": state.company.name,
            "company_sections": state.company.sections,
        },
        prior_verdicts=state.verdicts,
        retry_attempt=state.retry_attempt,
    )
    draft = await writer.invoke(ctx)
    if not isinstance(draft, KBArticleDraft):
        raise TypeError(f"Expected KBArticleDraft, got {type(draft).__name__}")
    return {"current_article_draft": draft}
```

Note: The placeholder prompt `phase1.problem_generator` works for now because all phase1 templates produce JSON for any schema. Plan 5 will swap to dedicated `phase1.article_writer` template.

- [ ] **Step 4: Verify** — pytest 1 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase1_kb/nodes.py tests/unit/test_phase1_nodes_article.py
git commit -m "feat: add article_writer node for Phase 1"
```

---

## Task 6: Persistence nodes

**Files:**
- Modify (append): `src/csfd/phases/phase1_kb/nodes.py`
- Test: `tests/unit/test_phase1_nodes_persist.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase1_nodes_persist.py`:
```python
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from csfd.phases.phase1_kb.nodes import persist_problem_node, persist_article_node
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBArticleDraft,
    KBState,
    ProblemDraft,
)
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRepo,
    ProblemRepo,
    RunRecord,
    RunRepo,
)


def _bootstrap(tmp_db_path: Path) -> tuple[Database, str]:
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


def _state(run_id: str, problem: CommittedProblem) -> KBState:
    return KBState(
        run_id=run_id, run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        scenarios=ScenarioCatalogue(),
        problems_committed=[problem],
    )


def test_persist_problem_node_writes_row(tmp_db_path: Path) -> None:
    db, run_id = _bootstrap(tmp_db_path)
    pc = CommittedProblem(
        id=str(uuid4()),
        draft=ProblemDraft(title="t", description="d", category="c", severity="low"),
        has_kb=False, coverage_reasoning="r", coverage_confidence="medium",
    )
    state = _state(run_id, pc)
    persist_problem_node(state, db=db, problem=pc)
    rows = ProblemRepo(db).list_for_run(run_id)
    assert len(rows) == 1
    assert rows[0].id == pc.id
    assert rows[0].has_kb is False


def test_persist_article_node_writes_row(tmp_db_path: Path) -> None:
    db, run_id = _bootstrap(tmp_db_path)
    pc = CommittedProblem(
        id=str(uuid4()),
        draft=ProblemDraft(title="t", description="d", category="c", severity="low"),
        has_kb=True, coverage_reasoning="r", coverage_confidence="high",
    )
    state = _state(run_id, pc)
    persist_problem_node(state, db=db, problem=pc)
    article = CommittedArticle(
        id=str(uuid4()), problem_id=pc.id,
        draft=KBArticleDraft(title="x", content_markdown="y",
                              troubleshooting_steps=[{"step": "a", "expected_result": "b"}]),
    )
    persist_article_node(state, db=db, article=article)
    got = KBArticleRepo(db).get_by_problem(pc.id)
    assert got is not None
    assert got.title == "x"
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Append to `src/csfd/phases/phase1_kb/nodes.py`**

```python
import json
from datetime import datetime, timezone

from csfd.storage.db import Database
from csfd.storage.repository import (
    KBArticleRecord,
    KBArticleRepo,
    ProblemRecord,
    ProblemRepo,
)
from csfd.utils.hashing import short_hash


def persist_problem_node(
    state: KBState,
    *,
    db: Database,
    problem: CommittedProblem,
) -> dict[str, Any]:
    repo = ProblemRepo(db)
    repo.create(ProblemRecord(
        id=problem.id,
        run_id=state.run_id,
        title=problem.draft.title,
        description=problem.draft.description,
        category=problem.draft.category,
        severity=problem.draft.severity,
        has_kb=problem.has_kb,
        coverage_reasoning=problem.coverage_reasoning,
        coverage_confidence=problem.coverage_confidence,
        metadata_json=json.dumps(problem.draft.metadata) if problem.draft.metadata else None,
        quality_flag=problem.quality_flag,
        unresolved_issues_json=(
            json.dumps([v.model_dump() for v in problem.unresolved_issues])
            if problem.unresolved_issues else None
        ),
        created_at=datetime.now(timezone.utc),
    ))
    return {"stats": state.stats.model_copy(update={
        "problems_committed": state.stats.problems_committed + 1,
    })}


def persist_article_node(
    state: KBState,
    *,
    db: Database,
    article: CommittedArticle,
) -> dict[str, Any]:
    repo = KBArticleRepo(db)
    repo.create(KBArticleRecord(
        id=article.id,
        run_id=state.run_id,
        problem_id=article.problem_id,
        title=article.draft.title,
        content_markdown=article.draft.content_markdown,
        content_hash=short_hash(article.draft.content_markdown),
        troubleshooting_steps_json=json.dumps(article.draft.troubleshooting_steps),
        prerequisites_json=(
            json.dumps(article.draft.prerequisites)
            if article.draft.prerequisites else None
        ),
        metadata_json=(
            json.dumps(article.draft.metadata) if article.draft.metadata else None
        ),
        version=1,
        quality_flag=article.quality_flag,
        unresolved_issues_json=(
            json.dumps([v.model_dump() for v in article.unresolved_issues])
            if article.unresolved_issues else None
        ),
        created_at=datetime.now(timezone.utc),
    ))
    return {"stats": state.stats.model_copy(update={
        "articles_committed": state.stats.articles_committed + 1,
    })}
```

- [ ] **Step 4: Verify** — pytest 2 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase1_kb/nodes.py tests/unit/test_phase1_nodes_persist.py
git commit -m "feat: add persist_problem + persist_article nodes for Phase 1"
```

---

## Task 7: Routing — `Send` fan-out + verdict aggregation

**Files:**
- Create: `src/csfd/phases/phase1_kb/routing.py`
- Test: `tests/unit/test_phase1_routing.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase1_routing.py`:
```python
from uuid import uuid4

from langgraph.types import Send

from csfd.agents.base import Issue, Verdict
from csfd.phases.phase1_kb.routing import (
    aggregate_verdicts,
    dispatch_problem_checkers,
    route_problem_verdict,
)
from csfd.phases.phase1_kb.state import KBState, ProblemDraft
from csfd.seeds.company import CompanyProfile
from csfd.seeds.scenarios import ScenarioCatalogue


def _state(*, verdicts: list[Verdict] | None = None, retry: int = 0) -> KBState:
    return KBState(
        run_id=str(uuid4()), run_seed=1,
        company=CompanyProfile(name="A", raw_markdown="# A"),
        scenarios=ScenarioCatalogue(),
        current_problem_draft=ProblemDraft(
            title="t", description="d", category="c", severity="low"
        ),
        verdicts=verdicts or [],
        retry_attempt=retry,
    )


def test_dispatch_problem_checkers_sends_to_three_nodes() -> None:
    state = _state()
    sends = dispatch_problem_checkers(state)
    targets = sorted(s.node for s in sends)
    assert targets == ["problem_background_check", "problem_consistency_check", "problem_scenario_check"]
    for s in sends:
        assert isinstance(s, Send)


def test_route_problem_verdict_pass_returns_advance() -> None:
    state = _state(verdicts=[
        Verdict(checker="consistency", passed=True),
        Verdict(checker="background", passed=True),
        Verdict(checker="scenario", passed=True),
    ])
    assert route_problem_verdict(state, max_retries=3) == "commit"


def test_route_problem_verdict_fail_under_budget_returns_regenerate() -> None:
    state = _state(
        verdicts=[
            Verdict(checker="consistency", passed=False, issues=[
                Issue(severity="error", location="x", rule_violated="r", explanation="e"),
            ]),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ],
        retry=1,
    )
    assert route_problem_verdict(state, max_retries=3) == "regenerate"


def test_route_problem_verdict_fail_over_budget_returns_commit_with_warning() -> None:
    state = _state(
        verdicts=[
            Verdict(checker="consistency", passed=False, issues=[
                Issue(severity="error", location="x", rule_violated="r", explanation="e"),
            ]),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ],
        retry=3,
    )
    assert route_problem_verdict(state, max_retries=3) == "commit_with_warning"


def test_aggregate_verdicts_is_noop_passthrough() -> None:
    """Aggregator just lets the state's accumulated verdicts flow through."""
    state = _state(verdicts=[Verdict(checker="x", passed=True)])
    update = aggregate_verdicts(state)
    assert update == {}
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Implement `src/csfd/phases/phase1_kb/routing.py`**

```python
"""Routing helpers for the Phase 1 LangGraph subgraph."""
from __future__ import annotations

from typing import Any

from langgraph.types import Send

from csfd.phases.phase1_kb.state import KBState


def dispatch_problem_checkers(state: KBState) -> list[Send]:
    """Fan out the candidate problem to the 3 checkers."""
    payload = {"current_problem_draft": state.current_problem_draft}
    return [
        Send("problem_consistency_check", payload),
        Send("problem_background_check", payload),
        Send("problem_scenario_check", payload),
    ]


def aggregate_verdicts(state: KBState) -> dict[str, Any]:
    """The Annotated[list, operator.add] reducer has already accumulated verdicts.
    This node is a no-op fan-in barrier — its presence makes the graph readable
    and provides a single seam for adding logic later (e.g. quorum voting).
    """
    return {}


def route_problem_verdict(state: KBState, *, max_retries: int) -> str:
    """Decide what happens after the 3 checkers' verdicts are aggregated."""
    if all(v.passed for v in state.verdicts):
        return "commit"
    if state.retry_attempt >= max_retries:
        return "commit_with_warning"
    return "regenerate"
```

- [ ] **Step 4: Verify** — pytest 5 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase1_kb/routing.py tests/unit/test_phase1_routing.py
git commit -m "feat: add Send-based dispatch and verdict routing for Phase 1"
```

---

## Task 8: Checker node wrappers

**Files:**
- Modify (append): `src/csfd/phases/phase1_kb/nodes.py`
- Test: `tests/unit/test_phase1_checker_nodes.py`

LangGraph nodes invoke checkers via `Send` payloads. Each wrapper takes a payload (dict) returned by `dispatch_problem_checkers`, builds the Checker via the factory, runs it, and returns a state update with one verdict.

- [ ] **Step 1: Failing test**

`tests/unit/test_phase1_checker_nodes.py`:
```python
from pathlib import Path

import pytest

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import problem_consistency_check_node
from csfd.phases.phase1_kb.state import ProblemDraft
from csfd.prompts.registry import PromptRegistry
from csfd.settings import AgentLLMConfig


def _factory(canned: Verdict) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={Verdict: canned}),
        agent_configs={
            "problem_consistency": AgentLLMConfig(
                provider="anthropic", model="claude-haiku-4-5", temperature=0.1,
            ),
        },
    )


@pytest.mark.asyncio
async def test_problem_consistency_check_returns_verdict_in_state() -> None:
    canned = Verdict(checker="consistency", passed=True)
    factory = _factory(canned)
    payload = {
        "current_problem_draft": ProblemDraft(
            title="t", description="d", category="c", severity="low",
        ),
    }
    update = await problem_consistency_check_node(payload, factory=factory)
    assert "verdicts" in update
    assert len(update["verdicts"]) == 1
    assert update["verdicts"][0].checker == "consistency"
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Append to `src/csfd/phases/phase1_kb/nodes.py`**

```python
async def _run_checker(
    *,
    factory: AgentFactory,
    name: str,
    prompt_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    checker = factory.build_checker(name=name, prompt_name=prompt_name)
    ctx = AgentContext(inputs=payload)
    verdict = await checker.invoke(ctx)
    return {"verdicts": [verdict]}


async def problem_consistency_check_node(
    payload: dict[str, Any],
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    return await _run_checker(
        factory=factory, name="problem_consistency",
        prompt_name="phase1.article_consistency",  # placeholder template
        payload=payload,
    )


async def problem_background_check_node(
    payload: dict[str, Any],
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    return await _run_checker(
        factory=factory, name="problem_background",
        prompt_name="phase1.article_consistency",  # placeholder
        payload=payload,
    )


async def problem_scenario_check_node(
    payload: dict[str, Any],
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    return await _run_checker(
        factory=factory, name="problem_scenario",
        prompt_name="phase1.article_consistency",  # placeholder
        payload=payload,
    )
```

(For Plan 3, all three placeholders point at the same template. Plan 5 will give each role a dedicated prompt.)

- [ ] **Step 4: Verify** — pytest 1 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase1_kb/nodes.py tests/unit/test_phase1_checker_nodes.py
git commit -m "feat: add Phase 1 checker node wrappers (consistency/background/scenario)"
```

---

## Task 9: `build_phase1_graph` subgraph builder

**Files:**
- Create: `src/csfd/phases/phase1_kb/subgraph.py`
- Test: `tests/unit/test_phase1_subgraph.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase1_subgraph.py`:
```python
import pytest

from csfd.phases.phase1_kb.subgraph import build_phase1_graph


def test_build_phase1_graph_compiles() -> None:
    graph = build_phase1_graph(max_retries=3, dedup_threshold=0.85, kb_target_rate=0.7)
    # Compiled graph has nodes and edges
    nodes = graph.get_graph().nodes
    expected = {
        "seed_load",
        "problem_brainstorm",
        "problem_check_dispatch",
        "problem_consistency_check",
        "problem_background_check",
        "problem_scenario_check",
        "problem_aggregate",
        "problem_route",
    }
    for n in expected:
        assert n in nodes


@pytest.mark.asyncio
async def test_build_phase1_graph_renders_mermaid_without_error() -> None:
    graph = build_phase1_graph(max_retries=3, dedup_threshold=0.85, kb_target_rate=0.7)
    text = graph.get_graph().draw_mermaid()
    assert "problem_brainstorm" in text
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Implement `src/csfd/phases/phase1_kb/subgraph.py`**

```python
"""Phase 1 LangGraph subgraph composition."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from csfd.phases.phase1_kb.nodes import (
    problem_background_check_node,
    problem_brainstorm_node,
    problem_consistency_check_node,
    problem_scenario_check_node,
    seed_load_node,
)
from csfd.phases.phase1_kb.routing import (
    aggregate_verdicts,
    dispatch_problem_checkers,
    route_problem_verdict,
)
from csfd.phases.phase1_kb.state import KBState


def build_phase1_graph(
    *,
    max_retries: int,
    dedup_threshold: float,  # noqa: ARG001  (used by extended graph in Plan 5)
    kb_target_rate: float,  # noqa: ARG001
) -> CompiledStateGraph:
    """Build (and compile) the minimal Phase 1 subgraph: brainstorm a problem,
    dispatch 3 parallel checkers, aggregate, route to commit / regenerate /
    commit_with_warning. Coverage decision, article writing, and persistence
    are wired up in tests directly rather than as graph nodes for now — Plan 5
    will compose the full multi-problem loop with the parent graph."""
    g = StateGraph(KBState)
    g.add_node("seed_load", seed_load_node)
    # NOTE: brainstorm/check nodes are async functions taking (state, **kwargs).
    # LangGraph injects only `state`; per-node kwargs (factory) are bound at
    # subgraph instantiation in Plan 5 via functools.partial. Here we use
    # placeholder lambdas that raise — production usage is via the parent graph.
    def _placeholder(_state: KBState) -> dict[str, object]:
        raise NotImplementedError(
            "Bind agent factories at subgraph instantiation (Plan 5)."
        )

    g.add_node("problem_brainstorm", _placeholder)
    g.add_node("problem_check_dispatch", _placeholder)
    g.add_node("problem_consistency_check", _placeholder)
    g.add_node("problem_background_check", _placeholder)
    g.add_node("problem_scenario_check", _placeholder)
    g.add_node("problem_aggregate", aggregate_verdicts)
    g.add_node("problem_route", _placeholder)

    g.add_edge(START, "seed_load")
    g.add_edge("seed_load", "problem_brainstorm")
    g.add_conditional_edges(
        "problem_brainstorm",
        dispatch_problem_checkers,
        {
            "problem_consistency_check": "problem_consistency_check",
            "problem_background_check": "problem_background_check",
            "problem_scenario_check": "problem_scenario_check",
        },
    )
    g.add_edge("problem_consistency_check", "problem_aggregate")
    g.add_edge("problem_background_check", "problem_aggregate")
    g.add_edge("problem_scenario_check", "problem_aggregate")

    def _route(state: KBState) -> str:
        return route_problem_verdict(state, max_retries=max_retries)

    g.add_conditional_edges(
        "problem_aggregate",
        _route,
        {
            "commit": END,
            "commit_with_warning": END,
            "regenerate": "problem_brainstorm",
        },
    )

    return g.compile()
```

NOTE: this Plan 3 build_phase1_graph is intentionally a *structural* subgraph — Plan 5 will produce a fully-wired graph that injects the `AgentFactory` and `Database` into the nodes via partials and adds the per-problem outer loop, coverage decider, article writer, and persistence. The integration test in Task 10 of THIS plan exercises the nodes directly (without the graph) to validate the runtime wiring.

- [ ] **Step 4: Verify** — pytest 2 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase1_kb/subgraph.py tests/unit/test_phase1_subgraph.py
git commit -m "feat: add Phase 1 subgraph structural composition"
```

---

## Task 10: End-to-end Phase 1 integration test (no graph; direct node composition)

**Files:**
- Test: `tests/integration/test_phase1_e2e_tiny.py`

This test composes the Phase 1 nodes directly (without the LangGraph graph) and demonstrates the full loop on tiny seeds: load → brainstorm → 3 checkers in parallel → aggregate → route → commit → coverage decide → write article → 3 article checkers → persist.

- [ ] **Step 1: Write the test**

```python
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.nodes import (
    article_writer_node,
    coverage_decider_node,
    persist_article_node,
    persist_problem_node,
    problem_background_check_node,
    problem_brainstorm_node,
    problem_consistency_check_node,
    problem_scenario_check_node,
)
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    CoverageDecision,
    KBArticleDraft,
    KBState,
    ProblemDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import parse_company_seed
from csfd.seeds.scenarios import parse_scenarios_seed
from csfd.settings import AgentLLMConfig
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRepo,
    ProblemRepo,
    RunRecord,
    RunRepo,
)


def _factory(canned: dict) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.5)
    agent_names = (
        "problem_brainstorm", "problem_consistency", "problem_background",
        "problem_scenario", "coverage_judge", "article_writer",
        "article_consistency", "article_background", "article_scenario",
    )
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured=canned),
        agent_configs={n: cfg for n in agent_names},
    )


@pytest.mark.asyncio
async def test_phase1_full_pipeline_on_tiny_seeds(tmp_db_path: Path) -> None:
    # Setup DB + run
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="running",
        started_at=datetime.now(timezone.utc), completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None, config_snapshot_json="{}",
        stats_json=None, error_summary=None,
    ))

    # Canned outputs for every agent
    canned_problem = ProblemDraft(
        title="Cannot reset password",
        description="Customer locked out after failed attempts",
        category="Authentication",
        severity="medium",
    )
    canned_decision = CoverageDecision(
        has_kb=True, reasoning="Common", confidence="high",
    )
    canned_article = KBArticleDraft(
        title="How to recover login",
        content_markdown="## Step 1\nReset via email link.",
        troubleshooting_steps=[
            {"step": "Click 'forgot password'", "expected_result": "email sent"},
        ],
    )
    canned_pass = Verdict(checker="x", passed=True)

    factory = _factory({
        ProblemDraft: canned_problem,
        CoverageDecision: canned_decision,
        KBArticleDraft: canned_article,
        Verdict: canned_pass,
    })

    # Build initial state
    state = KBState(
        run_id=run_id, run_seed=1,
        company=parse_company_seed(Path("tests/fixtures/tiny_company_seed.md")),
        scenarios=parse_scenarios_seed(Path("tests/fixtures/tiny_scenarios_seed.md")),
    )

    # 1. Brainstorm a candidate problem
    update = await problem_brainstorm_node(state, factory=factory)
    state = state.model_copy(update=update)
    assert state.current_problem_draft is not None

    # 2. Run 3 checkers in parallel
    payload = {"current_problem_draft": state.current_problem_draft}
    verdicts = await asyncio.gather(
        problem_consistency_check_node(payload, factory=factory),
        problem_background_check_node(payload, factory=factory),
        problem_scenario_check_node(payload, factory=factory),
    )
    state = state.model_copy(update={
        "verdicts": [v["verdicts"][0] for v in verdicts],
    })
    assert all(v.passed for v in state.verdicts)

    # 3. Commit problem (still uncovered at this point)
    pid = str(uuid4())
    pc = CommittedProblem(
        id=pid,
        draft=state.current_problem_draft,
        has_kb=False,  # will be set by coverage_decider
        coverage_reasoning="",
        coverage_confidence="low",
    )
    state = state.model_copy(update={
        "problems_committed": [pc],
        "verdicts": [],  # reset for next round
    })

    # 4. Coverage decision
    update = await coverage_decider_node(state, factory=factory, target_rate=1.0)
    state = state.model_copy(update=update)
    assert state.problems_committed[0].has_kb is True

    # 5. Persist problem
    persist_problem_node(state, db=db, problem=state.problems_committed[0])

    # 6. Write article
    update = await article_writer_node(
        state, factory=factory, problem=state.problems_committed[0],
    )
    state = state.model_copy(update=update)
    assert state.current_article_draft is not None

    # 7. Persist article
    article = CommittedArticle(
        id=str(uuid4()), problem_id=pid, draft=state.current_article_draft,
    )
    persist_article_node(state, db=db, article=article)

    # 8. Verify SQLite contents
    problems = ProblemRepo(db).list_for_run(run_id)
    assert len(problems) == 1
    assert problems[0].has_kb is True
    article_row = KBArticleRepo(db).get_by_problem(pid)
    assert article_row is not None
    assert article_row.title == "How to recover login"
```

- [ ] **Step 2: Verify** — `uv run pytest tests/integration/test_phase1_e2e_tiny.py -v` → 1 pass; `uv run pytest -q` → no regressions; mypy + ruff clean.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase1_e2e_tiny.py
git commit -m "test: end-to-end Phase 1 pipeline integration with FakeChatModel"
```

---

## Task 11: `phases/phase1_kb/__init__.py` re-exports

**Files:**
- Modify: `src/csfd/phases/phase1_kb/__init__.py`

- [ ] **Step 1: Write**

```python
"""csfd.phases.phase1_kb — Phase 1 KB-generation subgraph."""
from csfd.phases.phase1_kb.nodes import (
    article_writer_node,
    coverage_decider_node,
    dedup_node,
    persist_article_node,
    persist_problem_node,
    problem_background_check_node,
    problem_brainstorm_node,
    problem_consistency_check_node,
    problem_scenario_check_node,
    seed_load_node,
)
from csfd.phases.phase1_kb.routing import (
    aggregate_verdicts,
    dispatch_problem_checkers,
    route_problem_verdict,
)
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    CoverageDecision,
    KBArticleDraft,
    KBState,
    PhaseStats,
    ProblemDraft,
)
from csfd.phases.phase1_kb.subgraph import build_phase1_graph

__all__ = [
    "CommittedArticle",
    "CommittedProblem",
    "CoverageDecision",
    "KBArticleDraft",
    "KBState",
    "PhaseStats",
    "ProblemDraft",
    "aggregate_verdicts",
    "article_writer_node",
    "build_phase1_graph",
    "coverage_decider_node",
    "dedup_node",
    "dispatch_problem_checkers",
    "persist_article_node",
    "persist_problem_node",
    "problem_background_check_node",
    "problem_brainstorm_node",
    "problem_consistency_check_node",
    "problem_scenario_check_node",
    "route_problem_verdict",
    "seed_load_node",
]
```

- [ ] **Step 2: Verify** — `uv run python -c "from csfd.phases.phase1_kb import KBState; print('ok')"` → prints `ok`; full suite green.

- [ ] **Step 3: Commit**

```bash
git add src/csfd/phases/phase1_kb/__init__.py
git commit -m "feat: re-export Phase 1 symbols at csfd.phases.phase1_kb"
```

---

## Task 12: Full-suite verification

- [ ] **Step 1**: `uv run pytest -v` — all green
- [ ] **Step 2**: `uv run mypy src tests` — clean
- [ ] **Step 3**: `uv run ruff check src tests` + `uv run ruff format --check src tests` — clean
- [ ] **Step 4**: If ruff format made changes, commit:

```bash
git add -A
git commit -m "chore: ruff format after Plan 3 work"
```

- [ ] **Step 5**: Capture summary — `git log --oneline | head -20`, test count, source file count.

## Plan 3 completion criteria

- [ ] All 12 tasks complete; tests pass.
- [ ] `csfd.phases.phase1_kb` package exposes the listed symbols.
- [ ] End-to-end test demonstrates: seed_load → brainstorm → 3 parallel checkers → coverage_decide → article_write → persist, all with `FakeChatModel`.
- [ ] `build_phase1_graph` compiles structurally and renders mermaid.
- [ ] Full suite green; mypy strict clean; ruff clean.

Plan 3 done. Hand off to **Plan 4 — Phase 2 (Case Generation)**.
