# CSFD Phase 2 — Case Generation Implementation Plan (Plan 4 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Build the Phase 2 LangGraph subgraph that consumes a Phase 1 KB (problems + kb_articles in SQLite) and produces multi-turn customer-service tickets. After Plan 4: a runnable `phase2` subgraph that, given a `parent_run_id` and a `FakeChatModel`, loads the problem pool from the parent run, samples (problem, ticket_type) pairs honouring the hard `has_kb=False ⇒ L3` routing rule, generates turns iteratively with probabilistic creative-noise injection, runs the same 3-checker parallelization on each turn, and persists tickets + turns to SQLite. End-to-end integration test exercises both KB and no-KB paths and asserts noise stratification.

**Architecture:**
- `TicketState` is a Pydantic BaseModel mirroring `KBState`'s shape but for ticket/turn artifacts.
- `Annotated[list[Verdict], operator.add]` reducer fan-in (identical pattern to Phase 1).
- Three turn-checkers fan out via `Send`; verdicts aggregate via reducer; routing decides `commit | regenerate | commit_with_warning`.
- Creative noise is a probabilistic gate before the checkers — sampled via seeded RNG so reruns are reproducible.
- Hard routing rule: `problem.has_kb=False ⇒ ticket_type=L3`. Otherwise sample via `weighted_choice` against `Phase2Config.ticket_type_weights.has_kb`.
- Outer loop iterates over tickets-per-problem (uniform integer in `[min, max]`); inner loop iterates turns until `turn_loop_decider` returns `ticket_done`.
- Bounded retries via `max_retries_per_artifact`; on exhaustion: `commit_with_warning` (turn persisted with `quality_flag`).

**Tech Stack:** Builds on Plans 1+2+3. Reuses `AgentFactory`, `Verdict`/`Issue`, `lexical_dedup`-style patterns, all Phase 1 routing helpers. Adds: Phase 2 node functions, `Phase2Config`-driven sampling via `derive_rng` / `weighted_choice`.

**Working directory:** `/Users/joao.augusto/Documents/Customer Service Fake Data`

---

## File Structure (Plan 4 scope)

| Path | Responsibility |
|---|---|
| `src/csfd/phases/phase2_cases/state.py` | `TicketState`, `TicketDraft`, `TurnDraft`, persona models, `CommittedTicket`, `CommittedTurn`, `Phase2Stats` |
| `src/csfd/phases/phase2_cases/sampler.py` | `sample_ticket_type(problem, rng, weights)` deterministic; `sample_tickets_for_problem(...)` count sampler |
| `src/csfd/phases/phase2_cases/nodes.py` | Async node functions (load_kb, ticket_sampler, ticket_init, turn_writer, creative_noise, turn_checkers, persisters) |
| `src/csfd/phases/phase2_cases/routing.py` | `Send`-based dispatch + verdict aggregation + retry routing + turn-loop / ticket-loop deciders |
| `src/csfd/phases/phase2_cases/subgraph.py` | `build_phase2_graph(...)` — composes the StateGraph |
| `src/csfd/phases/phase2_cases/__init__.py` | re-exports |
| `tests/unit/test_phase2_state.py` | TicketState shape + reducer tests |
| `tests/unit/test_phase2_sampler.py` | sampler determinism + hard L3 rule tests |
| `tests/unit/test_phase2_nodes_load_kb.py` | `load_kb_node` reads parent run's problems + kb_articles |
| `tests/unit/test_phase2_nodes_ticket.py` | `ticket_sampler_node` + `ticket_init_node` tests |
| `tests/unit/test_phase2_nodes_turn.py` | `turn_writer_node` test |
| `tests/unit/test_phase2_nodes_noise.py` | `creative_noise_gate` + `creative_noise_node` tests |
| `tests/unit/test_phase2_checker_nodes.py` | Turn checker wrapper tests |
| `tests/unit/test_phase2_nodes_persist.py` | `persist_ticket_node` + `persist_turn_node` tests |
| `tests/unit/test_phase2_routing.py` | Send dispatch + verdict routing + turn_loop_decider tests |
| `tests/unit/test_phase2_subgraph.py` | Subgraph compiles + mermaid renders |
| `tests/integration/test_phase2_e2e_tiny.py` | Phase 2 e2e with FakeChatModel: both KB and no-KB paths, noise stratification |

Out of scope: real LLM integration tests (Plan 5 cassettes), parent graph composition with shared checkpointer (Plan 5), CLI commands (Plan 5).

---

## Task 1: `TicketState` + draft / persona models

**Files:**
- Create: `src/csfd/phases/phase2_cases/state.py`
- Test: `tests/unit/test_phase2_state.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_state.py`:
```python
from typing import get_args, get_type_hints
from uuid import uuid4

from csfd.agents.base import Verdict
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CommittedTurn,
    CustomerPersona,
    Phase2Stats,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.seeds.company import CompanyProfile


def _empty_company() -> CompanyProfile:
    return CompanyProfile(name="Acme", raw_markdown="# Acme")


def test_ticket_state_defaults() -> None:
    state = TicketState(
        run_id=str(uuid4()),
        parent_run_id=str(uuid4()),
        run_seed=42,
        company=_empty_company(),
    )
    assert state.problem_pool == []
    assert state.kb_by_problem == {}
    assert state.current_ticket is None
    assert state.turns_committed == []
    assert state.current_turn_draft is None
    assert state.noise_applied is False
    assert state.noise_type is None
    assert state.verdicts == []
    assert state.retry_attempt == 0
    assert state.turn_index == 0
    assert isinstance(state.stats, Phase2Stats)


def test_ticket_state_verdict_reducer_is_operator_add() -> None:
    import operator
    hints = get_type_hints(TicketState, include_extras=True)
    annotated = hints["verdicts"]
    args = get_args(annotated)
    assert args[1] is operator.add


def test_turn_draft_required_fields() -> None:
    t = TurnDraft(
        speaker="agent",
        content="Hello, how can I help?",
        intent="question",
    )
    assert t.speaker == "agent"
    assert t.kb_references == []
    assert t.noise_applied is False


def test_ticket_draft_required_fields() -> None:
    td = TicketDraft(
        problem_id="p1",
        kb_article_id=None,
        ticket_type="l3",
        priority="high",
        subject="Cannot log in",
        customer_persona=CustomerPersona(name="C", tier="standard", tone="frustrated"),
        agent_persona=AgentPersona(name="A", tier="l3", expertise="auth specialist"),
    )
    assert td.ticket_type == "l3"
    assert td.kb_article_id is None


def test_committed_ticket_and_turn() -> None:
    ct = CommittedTicket(
        id="t1",
        draft=TicketDraft(
            problem_id="p1",
            kb_article_id="a1",
            ticket_type="l1",
            priority="medium",
            subject="s",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="neutral"),
            agent_persona=AgentPersona(name="A", tier="l1", expertise=""),
        ),
        status="resolved",
    )
    assert ct.draft.ticket_type == "l1"

    cu = CommittedTurn(
        id="tu1",
        ticket_id="t1",
        turn_index=0,
        draft=TurnDraft(speaker="customer", content="help", intent="question"),
    )
    assert cu.turn_index == 0
```

- [ ] **Step 2: Confirm failure** — `uv run pytest tests/unit/test_phase2_state.py -v` → ImportError.

- [ ] **Step 3: Implement `src/csfd/phases/phase2_cases/state.py`**

```python
"""TicketState + draft/persona/stats Pydantic models for the Phase 2 subgraph."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from csfd.agents.base import Verdict
from csfd.phases.phase1_kb.state import CommittedArticle, CommittedProblem
from csfd.seeds.company import CompanyProfile


class CustomerPersona(BaseModel):
    """Customer-side persona attached to a ticket."""

    name: str
    tier: str
    tone: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentPersona(BaseModel):
    """Support-rep persona attached to a ticket."""

    name: str
    tier: str  # 'docs'|'l1'|'l2'|'l3'
    expertise: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TicketDraft(BaseModel):
    """An LLM- or sampler-produced candidate ticket before persistence."""

    problem_id: str
    kb_article_id: str | None
    ticket_type: Literal["docs_request", "l1", "l2", "l3"]
    priority: Literal["low", "medium", "high", "critical"]
    subject: str
    customer_persona: CustomerPersona
    agent_persona: AgentPersona
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnDraft(BaseModel):
    """An LLM-generated candidate turn before persistence."""

    speaker: Literal["customer", "agent", "system"]
    content: str
    intent: Literal[
        "question", "clarification", "resolution", "escalation", "thanks", "closing",
    ] | None = None
    speaker_persona: str | None = None
    kb_references: list[str] = Field(default_factory=list)
    noise_applied: bool = False
    noise_type: str | None = None


class Phase2Stats(BaseModel):
    tickets_committed: int = 0
    turns_committed: int = 0
    turns_with_noise: int = 0
    total_tokens: int = 0
    total_usd: float = 0.0


class CommittedTicket(BaseModel):
    """A ticket committed during the run."""

    id: str
    draft: TicketDraft
    status: Literal["resolved", "unresolved", "escalated"] = "resolved"
    ground_truth: dict[str, Any] | None = None
    quality_flag: str | None = None
    unresolved_issues: list[Verdict] = Field(default_factory=list)


class CommittedTurn(BaseModel):
    """A turn committed during the run."""

    id: str
    ticket_id: str
    turn_index: int
    draft: TurnDraft
    quality_flag: str | None = None
    unresolved_issues: list[Verdict] = Field(default_factory=list)


class TicketState(BaseModel):
    """In-memory working state for the Phase 2 LangGraph subgraph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    parent_run_id: str
    run_seed: int
    company: CompanyProfile

    problem_pool: list[CommittedProblem] = Field(default_factory=list)
    kb_by_problem: dict[str, CommittedArticle] = Field(default_factory=dict)

    current_ticket: CommittedTicket | None = None
    turns_committed: list[CommittedTurn] = Field(default_factory=list)
    current_turn_draft: TurnDraft | None = None

    noise_applied: bool = False
    noise_type: str | None = None
    turn_index: int = 0

    verdicts: Annotated[list[Verdict], operator.add] = Field(default_factory=list)
    retry_attempt: int = 0

    stats: Phase2Stats = Field(default_factory=Phase2Stats)
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_state.py -v` → 5 pass; `uv run mypy src tests` clean; `uv run ruff check src tests` clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/state.py tests/unit/test_phase2_state.py
git commit -m "feat: add TicketState and Phase 2 draft models"
```

---

## Task 2: Ticket sampler — hard L3 rule + weighted sampling

**Files:**
- Create: `src/csfd/phases/phase2_cases/sampler.py`
- Test: `tests/unit/test_phase2_sampler.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_sampler.py`:
```python
from csfd.phases.phase1_kb.state import CommittedProblem, ProblemDraft
from csfd.phases.phase2_cases.sampler import (
    sample_ticket_count,
    sample_ticket_type,
)
from csfd.ticket_types.definitions import TicketType
from csfd.utils.rng import derive_rng


def _problem(has_kb: bool) -> CommittedProblem:
    return CommittedProblem(
        id="p",
        draft=ProblemDraft(title="t", description="d", category="c", severity="low"),
        has_kb=has_kb,
        coverage_reasoning="",
        coverage_confidence="low",
    )


def test_sample_ticket_type_no_kb_forces_l3() -> None:
    rng = derive_rng(1, "x")
    weights_has_kb = {"docs_request": 0.10, "l1": 0.55, "l2": 0.25, "l3": 0.10}
    weights_no_kb = {"l3": 1.0}
    # 100 trials: every no_kb result must be L3
    for _ in range(100):
        t = sample_ticket_type(
            _problem(has_kb=False), rng,
            weights_has_kb=weights_has_kb, weights_no_kb=weights_no_kb,
        )
        assert t == TicketType.L3


def test_sample_ticket_type_has_kb_uses_weighted_distribution() -> None:
    rng = derive_rng(42, "dist-check")
    weights_has_kb = {"docs_request": 0.10, "l1": 0.55, "l2": 0.25, "l3": 0.10}
    weights_no_kb = {"l3": 1.0}
    seen: dict[str, int] = {"docs_request": 0, "l1": 0, "l2": 0, "l3": 0}
    for _ in range(1000):
        t = sample_ticket_type(
            _problem(has_kb=True), rng,
            weights_has_kb=weights_has_kb, weights_no_kb=weights_no_kb,
        )
        seen[t.value] += 1
    # L1 is dominant in default weights
    assert seen["l1"] > seen["l2"] > seen["docs_request"]
    assert seen["l1"] > 400  # ~55% of 1000 with healthy margin
    # All buckets sampled at least once
    assert all(v > 0 for v in seen.values())


def test_sample_ticket_type_deterministic_under_same_seed() -> None:
    weights_has_kb = {"l1": 1.0}
    weights_no_kb = {"l3": 1.0}
    p = _problem(has_kb=True)
    rng_a = derive_rng(7, "x")
    rng_b = derive_rng(7, "x")
    seq_a = [
        sample_ticket_type(p, rng_a, weights_has_kb=weights_has_kb,
                            weights_no_kb=weights_no_kb)
        for _ in range(50)
    ]
    seq_b = [
        sample_ticket_type(p, rng_b, weights_has_kb=weights_has_kb,
                            weights_no_kb=weights_no_kb)
        for _ in range(50)
    ]
    assert seq_a == seq_b


def test_sample_ticket_count_in_range() -> None:
    rng = derive_rng(1, "count")
    for _ in range(50):
        n = sample_ticket_count(rng, range_inclusive=(3, 5))
        assert 3 <= n <= 5


def test_sample_ticket_count_single_value_range() -> None:
    rng = derive_rng(1, "count")
    assert sample_ticket_count(rng, range_inclusive=(2, 2)) == 2
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Implement `src/csfd/phases/phase2_cases/sampler.py`**

```python
"""Deterministic, seeded samplers for Phase 2 ticket generation."""
from __future__ import annotations

from collections.abc import Mapping
from random import Random

from csfd.phases.phase1_kb.state import CommittedProblem
from csfd.ticket_types.definitions import TicketType
from csfd.utils.rng import weighted_choice


def sample_ticket_type(
    problem: CommittedProblem,
    rng: Random,
    *,
    weights_has_kb: Mapping[str, float],
    weights_no_kb: Mapping[str, float],
) -> TicketType:
    """Pick a ticket type for `problem`.

    Hard rule: when `problem.has_kb=False`, the result is always `L3` regardless
    of weights — the no-KB path is the entire reason Phase 1 sets the flag.
    Otherwise pick proportionally to `weights_has_kb`.
    """
    if not problem.has_kb:
        # honour weights_no_kb but in practice it's `{l3: 1.0}` by default
        picked = weighted_choice(weights_no_kb, rng)
        # Enforce the hard rule even if a user mis-configures weights_no_kb
        if picked != TicketType.L3.value:
            return TicketType.L3
        return TicketType(picked)
    return TicketType(weighted_choice(weights_has_kb, rng))


def sample_ticket_count(rng: Random, *, range_inclusive: tuple[int, int]) -> int:
    """Pick a uniform integer in [lo, hi] inclusive."""
    lo, hi = range_inclusive
    return rng.randint(lo, hi)
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_sampler.py -v` → 5 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/sampler.py tests/unit/test_phase2_sampler.py
git commit -m "feat: add Phase 2 ticket sampler with hard no-KB→L3 rule"
```

---

## Task 3: `load_kb_node` — read Phase 1 KB from SQLite

**Files:**
- Create: `src/csfd/phases/phase2_cases/nodes.py` (start)
- Test: `tests/unit/test_phase2_nodes_load_kb.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_nodes_load_kb.py`:
```python
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBArticleDraft,
    ProblemDraft,
)
from csfd.phases.phase2_cases.nodes import load_kb_node
from csfd.phases.phase2_cases.state import TicketState
from csfd.seeds.company import CompanyProfile
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRecord,
    KBArticleRepo,
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
)
from csfd.utils.hashing import short_hash


def _bootstrap_phase1(tmp_db_path: Path) -> tuple[Database, str]:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    parent_run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=parent_run_id, phase="phase1", parent_run_id=None, status="completed",
        started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
        run_seed=1, pipeline_version="0.1.0", git_sha=None,
        config_snapshot_json="{}", stats_json=None, error_summary=None,
    ))
    # Two problems: one has_kb=True, one has_kb=False
    p_with_kb = str(uuid4())
    p_no_kb = str(uuid4())
    for pid, has_kb in [(p_with_kb, True), (p_no_kb, False)]:
        ProblemRepo(db).create(ProblemRecord(
            id=pid, run_id=parent_run_id, title=f"P-{pid[:4]}", description="d",
            category="c", severity="low", has_kb=has_kb,
            coverage_reasoning="r", coverage_confidence="medium",
            metadata_json=None, quality_flag=None, unresolved_issues_json=None,
            created_at=datetime.now(timezone.utc),
        ))
    # One KB article attached to the covered problem
    KBArticleRepo(db).create(KBArticleRecord(
        id=str(uuid4()), run_id=parent_run_id, problem_id=p_with_kb,
        title="Article", content_markdown="## Steps", content_hash=short_hash("## Steps"),
        troubleshooting_steps_json="[]", prerequisites_json=None,
        metadata_json=None, version=1, quality_flag=None,
        unresolved_issues_json=None, created_at=datetime.now(timezone.utc),
    ))
    return db, parent_run_id


@pytest.mark.asyncio
async def test_load_kb_node_populates_pool_and_articles(tmp_db_path: Path) -> None:
    db, parent_run_id = _bootstrap_phase1(tmp_db_path)
    state = TicketState(
        run_id=str(uuid4()), parent_run_id=parent_run_id, run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
    )
    update = await load_kb_node(state, db=db)
    assert "problem_pool" in update
    assert "kb_by_problem" in update
    pool = update["problem_pool"]
    kbs = update["kb_by_problem"]
    assert isinstance(pool, list)
    assert isinstance(kbs, dict)
    assert len(pool) == 2
    # Exactly one of the two problems has an article in kb_by_problem
    assert len(kbs) == 1
    covered = [p for p in pool if p.has_kb]
    assert len(covered) == 1
    assert covered[0].id in kbs
```

- [ ] **Step 2: Confirm failure** — ImportError on `load_kb_node`.

- [ ] **Step 3: Implement `src/csfd/phases/phase2_cases/nodes.py`** (initial — only `load_kb_node`; later tasks append more nodes)

```python
"""Async LangGraph nodes for the Phase 2 case-generation subgraph."""
from __future__ import annotations

import json
from typing import Any

from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBArticleDraft,
    ProblemDraft,
)
from csfd.phases.phase2_cases.state import TicketState
from csfd.storage.db import Database
from csfd.storage.repository import KBArticleRepo, ProblemRepo


async def load_kb_node(state: TicketState, *, db: Database) -> dict[str, Any]:
    """Load problems + KB articles from the parent Phase 1 run."""
    problems = ProblemRepo(db).list_for_run(state.parent_run_id)
    pool: list[CommittedProblem] = []
    for p in problems:
        pool.append(CommittedProblem(
            id=p.id,
            draft=ProblemDraft(
                title=p.title,
                description=p.description,
                category=p.category,
                severity=p.severity,  # type: ignore[arg-type]
                metadata=(
                    json.loads(p.metadata_json) if p.metadata_json else {}
                ),
            ),
            has_kb=p.has_kb,
            coverage_reasoning=p.coverage_reasoning or "",
            coverage_confidence=p.coverage_confidence or "low",
            quality_flag=p.quality_flag,
        ))

    kb_by_problem: dict[str, CommittedArticle] = {}
    repo = KBArticleRepo(db)
    for cp in pool:
        if not cp.has_kb:
            continue
        row = repo.get_by_problem(cp.id)
        if row is None:
            continue
        kb_by_problem[cp.id] = CommittedArticle(
            id=row.id,
            problem_id=row.problem_id,
            draft=KBArticleDraft(
                title=row.title,
                content_markdown=row.content_markdown,
                troubleshooting_steps=(
                    json.loads(row.troubleshooting_steps_json)
                    if row.troubleshooting_steps_json else []
                ),
                prerequisites=(
                    json.loads(row.prerequisites_json) if row.prerequisites_json else []
                ),
                metadata=(
                    json.loads(row.metadata_json) if row.metadata_json else {}
                ),
            ),
            quality_flag=row.quality_flag,
        )

    return {"problem_pool": pool, "kb_by_problem": kb_by_problem}
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_nodes_load_kb.py -v` → 1 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/nodes.py tests/unit/test_phase2_nodes_load_kb.py
git commit -m "feat: add load_kb_node for Phase 2 (loads parent run's problems + articles)"
```

---

## Task 4: `ticket_sampler_node` + `ticket_init_node`

**Files:**
- Modify (append): `src/csfd/phases/phase2_cases/nodes.py`
- Test: `tests/unit/test_phase2_nodes_ticket.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_nodes_ticket.py`:
```python
from uuid import uuid4

import pytest

from csfd.phases.phase1_kb.state import CommittedProblem, ProblemDraft
from csfd.phases.phase2_cases.nodes import ticket_init_node, ticket_sampler_node
from csfd.phases.phase2_cases.state import TicketState
from csfd.seeds.company import CompanyProfile
from csfd.settings import Phase2Config, TicketsPerProblem, TicketTypeWeights


def _phase2_cfg() -> Phase2Config:
    return Phase2Config(
        tickets_per_problem=TicketsPerProblem(has_kb=(2, 2), no_kb=(1, 1)),
        ticket_type_weights=TicketTypeWeights(
            has_kb={"docs_request": 0.0, "l1": 1.0, "l2": 0.0, "l3": 0.0},
            no_kb={"l3": 1.0},
        ),
        creative_noise_probability=0.0,
        noise_type_weights={"x": 1.0},
        min_turns_per_ticket=2,
        max_turns_per_ticket=4,
    )


def _state(problems: list[CommittedProblem]) -> TicketState:
    return TicketState(
        run_id=str(uuid4()),
        parent_run_id=str(uuid4()),
        run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        problem_pool=problems,
    )


@pytest.mark.asyncio
async def test_ticket_sampler_picks_l3_when_no_kb() -> None:
    p_no = CommittedProblem(
        id="p-no", draft=ProblemDraft(title="t", description="d", category="c",
                                      severity="low"),
        has_kb=False, coverage_reasoning="", coverage_confidence="low",
    )
    state = _state([p_no])
    update = await ticket_sampler_node(state, cfg=_phase2_cfg())
    assert "current_problem_id" in update
    assert update["current_problem_id"] == "p-no"
    assert update["ticket_type"] == "l3"


@pytest.mark.asyncio
async def test_ticket_sampler_picks_l1_when_has_kb_under_l1_weights() -> None:
    p_yes = CommittedProblem(
        id="p-yes", draft=ProblemDraft(title="t", description="d", category="c",
                                       severity="low"),
        has_kb=True, coverage_reasoning="r", coverage_confidence="high",
    )
    state = _state([p_yes])
    update = await ticket_sampler_node(state, cfg=_phase2_cfg())
    assert update["ticket_type"] == "l1"


@pytest.mark.asyncio
async def test_ticket_init_node_builds_ticket_draft_with_personas() -> None:
    p_yes = CommittedProblem(
        id="p-yes", draft=ProblemDraft(title="Cannot log in", description="d",
                                       category="auth", severity="medium"),
        has_kb=True, coverage_reasoning="r", coverage_confidence="high",
    )
    state = _state([p_yes])
    update = await ticket_init_node(
        state, problem=p_yes, ticket_type="l1", kb_article_id="a-1",
    )
    assert "current_ticket" in update
    ticket = update["current_ticket"]
    assert ticket.draft.problem_id == "p-yes"
    assert ticket.draft.ticket_type == "l1"
    assert ticket.draft.kb_article_id == "a-1"
    assert ticket.draft.customer_persona.tier in {"standard", "premium", "enterprise"}
    assert ticket.draft.agent_persona.tier == "l1"
    assert ticket.draft.subject  # non-empty
```

- [ ] **Step 2: Confirm failure** — ImportError on `ticket_sampler_node` / `ticket_init_node`.

- [ ] **Step 3: Append to `src/csfd/phases/phase2_cases/nodes.py`**

```python
from csfd.phases.phase1_kb.state import CommittedProblem  # already imported above
from csfd.phases.phase2_cases.sampler import (
    sample_ticket_count,
    sample_ticket_type,
)
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CustomerPersona,
    TicketDraft,
)
from csfd.settings import Phase2Config
from csfd.ticket_types.definitions import TICKET_TYPE_METADATA, TicketType
from csfd.utils.rng import derive_rng


async def ticket_sampler_node(
    state: TicketState, *, cfg: Phase2Config,
) -> dict[str, Any]:
    """Sample the next (problem, ticket_type) pair for this run.

    Honours the hard `has_kb=False ⇒ L3` rule via `sample_ticket_type`.
    For Plan 4 the sampler picks the *first* problem in the pool that still has
    capacity (Plan 5 will wire the outer loop via a counter on state)."""
    if not state.problem_pool:
        raise ValueError("ticket_sampler_node called with empty problem_pool")
    rng = derive_rng(state.run_seed, f"ticket_sampler:{len(state.turns_committed)}")
    problem = state.problem_pool[0]
    tt = sample_ticket_type(
        problem, rng,
        weights_has_kb=cfg.ticket_type_weights.has_kb,
        weights_no_kb=cfg.ticket_type_weights.no_kb,
    )
    return {
        "current_problem_id": problem.id,
        "ticket_type": tt.value,
    }


async def ticket_init_node(
    state: TicketState,
    *,
    problem: CommittedProblem,
    ticket_type: str,
    kb_article_id: str | None,
) -> dict[str, Any]:
    """Build a `CommittedTicket` (draft form) with synthesised personas."""
    tt = TicketType(ticket_type)
    md = TICKET_TYPE_METADATA[tt]
    rng = derive_rng(state.run_seed, f"persona:{problem.id}:{ticket_type}")
    customer_tier = rng.choice(["standard", "premium", "enterprise"])
    customer_tone = rng.choice(["neutral", "frustrated", "polite", "urgent"])
    ticket = CommittedTicket(
        id=problem.id + ":" + ticket_type + ":" + str(rng.randint(0, 1_000_000)),
        draft=TicketDraft(
            problem_id=problem.id,
            kb_article_id=kb_article_id,
            ticket_type=tt.value,  # type: ignore[arg-type]
            priority=("high" if tt == TicketType.L3 else "medium"),  # type: ignore[arg-type]
            subject=problem.draft.title,
            customer_persona=CustomerPersona(
                name=f"Customer-{rng.randint(1000, 9999)}",
                tier=customer_tier,
                tone=customer_tone,
            ),
            agent_persona=AgentPersona(
                name=f"Agent-{rng.randint(1000, 9999)}",
                tier=tt.value,
                expertise=md.persona_label,
            ),
        ),
    )
    return {"current_ticket": ticket, "turn_index": 0}


# Helper retained for callers that want the count without going through the node
def pick_ticket_count(
    state: TicketState, problem: CommittedProblem, cfg: Phase2Config,
) -> int:
    rng = derive_rng(state.run_seed, f"ticket_count:{problem.id}")
    range_inclusive = (
        cfg.tickets_per_problem.has_kb if problem.has_kb
        else cfg.tickets_per_problem.no_kb
    )
    return sample_ticket_count(rng, range_inclusive=range_inclusive)
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_nodes_ticket.py -v` → 3 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/nodes.py tests/unit/test_phase2_nodes_ticket.py
git commit -m "feat: add ticket_sampler + ticket_init nodes for Phase 2"
```

---

## Task 5: `turn_writer_node`

**Files:**
- Modify (append): `src/csfd/phases/phase2_cases/nodes.py`
- Test: `tests/unit/test_phase2_nodes_turn.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_nodes_turn.py`:
```python
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBArticleDraft,
    ProblemDraft,
)
from csfd.phases.phase2_cases.nodes import turn_writer_node
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CustomerPersona,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.settings import AgentLLMConfig


def _factory(canned: TurnDraft) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={TurnDraft: canned}),
        agent_configs={
            "turn_writer": AgentLLMConfig(
                provider="anthropic", model="claude-haiku-4-5", temperature=0.7,
            ),
        },
    )


def _state(with_kb: bool) -> TicketState:
    pid = "p-x"
    problem = CommittedProblem(
        id=pid,
        draft=ProblemDraft(title="Login fails", description="401 returned",
                           category="auth", severity="medium"),
        has_kb=with_kb,
        coverage_reasoning="r", coverage_confidence="high",
    )
    kbs: dict[str, CommittedArticle] = {}
    if with_kb:
        kbs[pid] = CommittedArticle(
            id="a-1", problem_id=pid,
            draft=KBArticleDraft(
                title="Recover login",
                content_markdown="## Step 1",
                troubleshooting_steps=[{"step": "x", "expected_result": "y"}],
            ),
        )
    ticket = CommittedTicket(
        id="t-1",
        draft=TicketDraft(
            problem_id=pid,
            kb_article_id="a-1" if with_kb else None,
            ticket_type="l3" if not with_kb else "l1",
            priority="medium",
            subject="Login fails",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="neutral"),
            agent_persona=AgentPersona(name="A", tier="l3" if not with_kb else "l1",
                                       expertise=""),
        ),
    )
    return TicketState(
        run_id=str(uuid4()), parent_run_id=str(uuid4()), run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
        problem_pool=[problem],
        kb_by_problem=kbs,
        current_ticket=ticket,
        turn_index=0,
    )


@pytest.mark.asyncio
async def test_turn_writer_emits_draft_into_state_with_kb() -> None:
    canned = TurnDraft(
        speaker="agent", content="Hi, please reset via email link.",
        intent="resolution",
    )
    factory = _factory(canned)
    state = _state(with_kb=True)
    update = await turn_writer_node(state, factory=factory)
    assert "current_turn_draft" in update
    draft = update["current_turn_draft"]
    assert isinstance(draft, TurnDraft)
    assert draft.speaker == "agent"


@pytest.mark.asyncio
async def test_turn_writer_runs_for_l3_no_kb_path() -> None:
    canned = TurnDraft(
        speaker="agent", content="Let me diagnose. Can you share recent logs?",
        intent="clarification",
    )
    factory = _factory(canned)
    state = _state(with_kb=False)
    update = await turn_writer_node(state, factory=factory)
    assert update["current_turn_draft"].intent == "clarification"
```

- [ ] **Step 2: Confirm failure** — ImportError on `turn_writer_node`.

- [ ] **Step 3: Append to `src/csfd/phases/phase2_cases/nodes.py`**

```python
from csfd.agents.base import AgentContext
from csfd.agents.factory import AgentFactory
from csfd.phases.phase2_cases.state import TurnDraft


async def turn_writer_node(
    state: TicketState,
    *,
    factory: AgentFactory,
) -> dict[str, Any]:
    """Generate the next turn for `state.current_ticket`.

    Reads the ticket's persona + prior turns + KB article (if any) so the
    Generator can produce a context-aware next turn.
    """
    if state.current_ticket is None:
        raise ValueError("turn_writer_node called without current_ticket")
    ticket = state.current_ticket
    pid = ticket.draft.problem_id
    article = state.kb_by_problem.get(pid)

    writer = factory.build_generator(
        name="turn_writer",
        prompt_name="phase2.turn_generator",
        output_schema_factory=lambda: TurnDraft,
    )
    ctx = AgentContext(
        inputs={
            "company_name": state.company.name,
            "ticket": ticket.draft.model_dump(),
            "kb_article": article.draft.model_dump() if article else None,
            "prior_turns": [
                {"speaker": t.draft.speaker, "content": t.draft.content,
                 "intent": t.draft.intent}
                for t in state.turns_committed
            ],
            "turn_index": state.turn_index,
        },
        prior_verdicts=state.verdicts,
        retry_attempt=state.retry_attempt,
    )
    draft = await writer.invoke(ctx)
    if not isinstance(draft, TurnDraft):
        raise TypeError(f"Expected TurnDraft, got {type(draft).__name__}")
    return {"current_turn_draft": draft}
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_nodes_turn.py -v` → 2 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/nodes.py tests/unit/test_phase2_nodes_turn.py
git commit -m "feat: add turn_writer node for Phase 2"
```

---

## Task 6: Creative noise — gate + noise node

**Files:**
- Modify (append): `src/csfd/phases/phase2_cases/nodes.py`
- Test: `tests/unit/test_phase2_nodes_noise.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_nodes_noise.py`:
```python
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.creative_noise import TurnModification
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase2_cases.nodes import (
    creative_noise_gate,
    creative_noise_node,
)
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CustomerPersona,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.settings import AgentLLMConfig


def _state(seed: int, *, with_turn: bool = True) -> TicketState:
    draft = TurnDraft(speaker="customer", content="help me", intent="question")
    ticket = CommittedTicket(
        id="t-1",
        draft=TicketDraft(
            problem_id="p-1", kb_article_id=None, ticket_type="l3",
            priority="medium", subject="x",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="n"),
            agent_persona=AgentPersona(name="A", tier="l3", expertise=""),
        ),
    )
    return TicketState(
        run_id=str(uuid4()), parent_run_id=str(uuid4()), run_seed=seed,
        company=CompanyProfile(name="Acme", raw_markdown="# A"),
        current_ticket=ticket,
        current_turn_draft=draft if with_turn else None,
        turn_index=0,
    )


def test_creative_noise_gate_returns_noise_when_under_probability() -> None:
    state = _state(seed=1)
    # probability=1.0 forces every call to enter the noise branch
    decision = creative_noise_gate(state, probability=1.0)
    assert decision == "apply_noise"


def test_creative_noise_gate_returns_skip_when_zero_probability() -> None:
    state = _state(seed=1)
    decision = creative_noise_gate(state, probability=0.0)
    assert decision == "skip_noise"


def test_creative_noise_gate_is_deterministic_under_same_seed() -> None:
    s1 = _state(seed=42)
    s2 = _state(seed=42)
    # State is identical, so the gate must produce the same decision.
    assert creative_noise_gate(s1, probability=0.5) == creative_noise_gate(s2, probability=0.5)


@pytest.mark.asyncio
async def test_creative_noise_node_applies_modification() -> None:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    canned = TurnModification(
        modified_content="help me!!!1 i waited so long :(",
        noise_type="typos_informal_phrasing",
        rationale="customer frustration",
    )
    factory = AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={TurnModification: canned}),
        agent_configs={
            "creative_noise": AgentLLMConfig(
                provider="anthropic", model="claude-haiku-4-5", temperature=0.95,
            ),
        },
    )
    state = _state(seed=1)
    update = await creative_noise_node(
        state, factory=factory,
        noise_type_weights={"typos_informal_phrasing": 1.0},
    )
    assert update["noise_applied"] is True
    assert update["noise_type"] == "typos_informal_phrasing"
    assert update["current_turn_draft"].content == canned.modified_content
    assert update["current_turn_draft"].noise_applied is True
    assert update["current_turn_draft"].noise_type == "typos_informal_phrasing"
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Append to `src/csfd/phases/phase2_cases/nodes.py`**

```python
from collections.abc import Mapping

from csfd.agents.creative_noise import TurnModification
from csfd.utils.rng import weighted_choice  # already imported indirectly; alias here for clarity


def creative_noise_gate(state: TicketState, *, probability: float) -> str:
    """Deterministic per-turn coin flip against `probability`.

    Returns `"apply_noise"` if the seeded RNG draw is below `probability`,
    otherwise `"skip_noise"`. The label is derived from `(run_seed, turn_index,
    current_ticket.id)` so each gate decision is uniquely seeded but
    reproducible.
    """
    if state.current_ticket is None:
        return "skip_noise"
    label = f"noise_gate:{state.current_ticket.id}:{state.turn_index}"
    rng = derive_rng(state.run_seed, label)
    draw = rng.random()
    return "apply_noise" if draw < probability else "skip_noise"


async def creative_noise_node(
    state: TicketState,
    *,
    factory: AgentFactory,
    noise_type_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Apply CreativeNoise to the current turn draft.

    Samples a noise type from `noise_type_weights`, invokes the `creative_noise`
    agent for the modified content, and returns a state update that overwrites
    `current_turn_draft` with the modified version and flips `noise_applied`/`
    noise_type` so persistence records the noise category.
    """
    if state.current_turn_draft is None:
        raise ValueError("creative_noise_node called without current_turn_draft")
    label = f"noise_type:{state.current_ticket.id if state.current_ticket else 'x'}:{state.turn_index}"
    rng = derive_rng(state.run_seed, label)
    noise_type = weighted_choice(noise_type_weights, rng)

    noise_agent = factory.build_creative_noise(
        name="creative_noise",
        prompt_name="phase2.creative_noise",
    )
    ctx = AgentContext(inputs={
        "turn_draft": state.current_turn_draft.model_dump(),
        "noise_type": noise_type,
    })
    mod = await noise_agent.invoke(ctx)
    if not isinstance(mod, TurnModification):
        raise TypeError(f"Expected TurnModification, got {type(mod).__name__}")
    modified = state.current_turn_draft.model_copy(update={
        "content": mod.modified_content,
        "noise_applied": True,
        "noise_type": mod.noise_type or noise_type,
    })
    return {
        "current_turn_draft": modified,
        "noise_applied": True,
        "noise_type": mod.noise_type or noise_type,
    }
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_nodes_noise.py -v` → 4 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/nodes.py tests/unit/test_phase2_nodes_noise.py
git commit -m "feat: add creative_noise gate + noise node for Phase 2"
```

---

## Task 7: Turn checker wrappers (3 nodes)

**Files:**
- Modify (append): `src/csfd/phases/phase2_cases/nodes.py`
- Test: `tests/unit/test_phase2_checker_nodes.py`

LangGraph nodes invoke checkers via `Send` payloads (mirrors Phase 1 pattern). Each wrapper takes a payload dict, builds the Checker via the factory, runs it, and returns a state update with one verdict.

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_checker_nodes.py`:
```python
from pathlib import Path

import pytest

from csfd.agents.base import Verdict
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase2_cases.nodes import (
    turn_background_check_node,
    turn_consistency_check_node,
    turn_scenario_check_node,
)
from csfd.phases.phase2_cases.state import TurnDraft
from csfd.prompts.registry import PromptRegistry
from csfd.settings import AgentLLMConfig


def _factory(canned: Verdict) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5",
                         temperature=0.1)
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={Verdict: canned}),
        agent_configs={
            "turn_consistency": cfg,
            "turn_background": cfg,
            "turn_scenario": cfg,
        },
    )


@pytest.mark.asyncio
async def test_turn_consistency_check_returns_verdict() -> None:
    canned = Verdict(checker="consistency", passed=True)
    factory = _factory(canned)
    payload = {"current_turn_draft": TurnDraft(
        speaker="agent", content="hello", intent="question",
    )}
    update = await turn_consistency_check_node(payload, factory=factory)
    assert update["verdicts"][0].checker == "consistency"


@pytest.mark.asyncio
async def test_turn_background_check_returns_verdict() -> None:
    canned = Verdict(checker="background", passed=False)
    factory = _factory(canned)
    payload = {"current_turn_draft": TurnDraft(
        speaker="agent", content="hello", intent="question",
    )}
    update = await turn_background_check_node(payload, factory=factory)
    assert update["verdicts"][0].checker == "background"
    assert update["verdicts"][0].passed is False


@pytest.mark.asyncio
async def test_turn_scenario_check_returns_verdict() -> None:
    canned = Verdict(checker="scenario", passed=True)
    factory = _factory(canned)
    payload = {"current_turn_draft": TurnDraft(
        speaker="agent", content="hello", intent="question",
    )}
    update = await turn_scenario_check_node(payload, factory=factory)
    assert update["verdicts"][0].checker == "scenario"
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Append to `src/csfd/phases/phase2_cases/nodes.py`**

```python
async def _run_turn_checker(
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


async def turn_consistency_check_node(
    payload: dict[str, Any], *, factory: AgentFactory,
) -> dict[str, Any]:
    return await _run_turn_checker(
        factory=factory, name="turn_consistency",
        prompt_name="phase2.turn_consistency", payload=payload,
    )


async def turn_background_check_node(
    payload: dict[str, Any], *, factory: AgentFactory,
) -> dict[str, Any]:
    return await _run_turn_checker(
        factory=factory, name="turn_background",
        prompt_name="phase2.turn_consistency", payload=payload,
    )


async def turn_scenario_check_node(
    payload: dict[str, Any], *, factory: AgentFactory,
) -> dict[str, Any]:
    return await _run_turn_checker(
        factory=factory, name="turn_scenario",
        prompt_name="phase2.turn_consistency", payload=payload,
    )
```

(For Plan 4 all three checkers point at the same placeholder template `phase2.turn_consistency`. Plan 5 will give each role its own dedicated prompt.)

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_checker_nodes.py -v` → 3 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/nodes.py tests/unit/test_phase2_checker_nodes.py
git commit -m "feat: add Phase 2 turn checker node wrappers"
```

---

## Task 8: Persistence nodes (`persist_ticket_node` + `persist_turn_node`)

**Files:**
- Modify (append): `src/csfd/phases/phase2_cases/nodes.py`
- Test: `tests/unit/test_phase2_nodes_persist.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_nodes_persist.py`:
```python
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from csfd.phases.phase1_kb.state import CommittedProblem, ProblemDraft
from csfd.phases.phase2_cases.nodes import (
    persist_ticket_node,
    persist_turn_node,
)
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CommittedTurn,
    CustomerPersona,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.seeds.company import CompanyProfile
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
    TurnRepo,
)


def _bootstrap(tmp_db_path: Path) -> tuple[Database, str, str]:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    parent_run_id = str(uuid4())
    run_id = str(uuid4())
    for rid, phase in [(parent_run_id, "phase1"), (run_id, "phase2")]:
        RunRepo(db).create(RunRecord(
            id=rid, phase=phase,
            parent_run_id=parent_run_id if phase == "phase2" else None,
            status="running", started_at=datetime.now(timezone.utc),
            completed_at=None, run_seed=1, pipeline_version="0.1.0",
            git_sha=None, config_snapshot_json="{}", stats_json=None,
            error_summary=None,
        ))
    pid = str(uuid4())
    ProblemRepo(db).create(ProblemRecord(
        id=pid, run_id=parent_run_id, title="t", description="d",
        category="c", severity="low", has_kb=False,
        coverage_reasoning="r", coverage_confidence="low",
        metadata_json=None, quality_flag=None, unresolved_issues_json=None,
        created_at=datetime.now(timezone.utc),
    ))
    return db, run_id, pid


def _state(run_id: str, ticket: CommittedTicket) -> TicketState:
    return TicketState(
        run_id=run_id, parent_run_id=str(uuid4()), run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# A"),
        current_ticket=ticket,
    )


def test_persist_ticket_node_writes_row(tmp_db_path: Path) -> None:
    db, run_id, pid = _bootstrap(tmp_db_path)
    ticket = CommittedTicket(
        id=str(uuid4()),
        draft=TicketDraft(
            problem_id=pid, kb_article_id=None, ticket_type="l3",
            priority="high", subject="x",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="n"),
            agent_persona=AgentPersona(name="A", tier="l3", expertise=""),
        ),
    )
    state = _state(run_id, ticket)
    persist_ticket_node(state, db=db, ticket=ticket)
    # Idempotency: a second call is a no-op
    persist_ticket_node(state, db=db, ticket=ticket)
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, ticket_type FROM tickets WHERE id = ?", (ticket.id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["ticket_type"] == "l3"


def test_persist_turn_node_writes_row(tmp_db_path: Path) -> None:
    db, run_id, pid = _bootstrap(tmp_db_path)
    ticket = CommittedTicket(
        id=str(uuid4()),
        draft=TicketDraft(
            problem_id=pid, kb_article_id=None, ticket_type="l3",
            priority="high", subject="x",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="n"),
            agent_persona=AgentPersona(name="A", tier="l3", expertise=""),
        ),
    )
    state = _state(run_id, ticket)
    persist_ticket_node(state, db=db, ticket=ticket)

    turn = CommittedTurn(
        id=str(uuid4()), ticket_id=ticket.id, turn_index=0,
        draft=TurnDraft(speaker="customer", content="help",
                        intent="question", noise_applied=True,
                        noise_type="typos_informal_phrasing"),
    )
    persist_turn_node(state, db=db, turn=turn)
    rows = TurnRepo(db).list_for_ticket(ticket.id)
    assert len(rows) == 1
    assert rows[0].turn_index == 0
    assert rows[0].noise_applied is True
    assert rows[0].noise_type == "typos_informal_phrasing"
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Append to `src/csfd/phases/phase2_cases/nodes.py`**

```python
import json
from datetime import datetime, timezone

from csfd.phases.phase2_cases.state import CommittedTurn
from csfd.storage.db import Database
from csfd.storage.repository import (
    TicketRecord,
    TicketRepo,
    TurnRecord,
    TurnRepo,
)


def persist_ticket_node(
    state: TicketState, *, db: Database, ticket: CommittedTicket,
) -> dict[str, Any]:
    """Idempotently persist a committed ticket row."""
    repo = TicketRepo(db)
    repo.create(TicketRecord(
        id=ticket.id,
        run_id=state.run_id,
        problem_id=ticket.draft.problem_id,
        kb_article_id=ticket.draft.kb_article_id,
        ticket_type=ticket.draft.ticket_type,
        priority=ticket.draft.priority,
        status=ticket.status,
        subject=ticket.draft.subject,
        customer_persona_json=ticket.draft.customer_persona.model_dump_json(),
        agent_persona_json=ticket.draft.agent_persona.model_dump_json(),
        ground_truth_json=(json.dumps(ticket.ground_truth)
                           if ticket.ground_truth else None),
        metadata_json=(json.dumps(ticket.draft.metadata)
                       if ticket.draft.metadata else None),
        quality_flag=ticket.quality_flag,
        unresolved_issues_json=(
            json.dumps([v.model_dump() for v in ticket.unresolved_issues])
            if ticket.unresolved_issues else None
        ),
        created_at=datetime.now(timezone.utc),
        resolved_at=None,
    ))
    return {"stats": state.stats.model_copy(update={
        "tickets_committed": state.stats.tickets_committed + 1,
    })}


def persist_turn_node(
    state: TicketState, *, db: Database, turn: CommittedTurn,
) -> dict[str, Any]:
    """Idempotently persist a committed turn row."""
    repo = TurnRepo(db)
    repo.create(TurnRecord(
        id=turn.id,
        ticket_id=turn.ticket_id,
        turn_index=turn.turn_index,
        speaker=turn.draft.speaker,
        speaker_persona=turn.draft.speaker_persona,
        content=turn.draft.content,
        intent=turn.draft.intent,
        kb_references_json=(
            json.dumps(turn.draft.kb_references)
            if turn.draft.kb_references else None
        ),
        noise_applied=turn.draft.noise_applied,
        noise_type=turn.draft.noise_type,
        quality_flag=turn.quality_flag,
        created_at=datetime.now(timezone.utc),
    ))
    updates: dict[str, Any] = {
        "stats": state.stats.model_copy(update={
            "turns_committed": state.stats.turns_committed + 1,
            "turns_with_noise": state.stats.turns_with_noise + (
                1 if turn.draft.noise_applied else 0
            ),
        }),
    }
    return updates
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_nodes_persist.py -v` → 2 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/nodes.py tests/unit/test_phase2_nodes_persist.py
git commit -m "feat: add persist_ticket + persist_turn nodes for Phase 2"
```

---

## Task 9: Routing — Send dispatch + verdict aggregation + turn-loop decider

**Files:**
- Create: `src/csfd/phases/phase2_cases/routing.py`
- Test: `tests/unit/test_phase2_routing.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_routing.py`:
```python
from uuid import uuid4

from langgraph.types import Send

from csfd.agents.base import Issue, Verdict
from csfd.phases.phase2_cases.routing import (
    aggregate_turn_verdicts,
    dispatch_turn_checkers,
    route_turn_loop,
    route_turn_verdict,
)
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CustomerPersona,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.seeds.company import CompanyProfile


def _state(
    *,
    verdicts: list[Verdict] | None = None,
    retry: int = 0,
    turn_index: int = 0,
    turns_committed: int = 0,
) -> TicketState:
    from csfd.phases.phase2_cases.state import CommittedTurn
    ticket = CommittedTicket(
        id="t-1",
        draft=TicketDraft(
            problem_id="p-1", kb_article_id=None, ticket_type="l1",
            priority="medium", subject="s",
            customer_persona=CustomerPersona(name="C", tier="standard", tone="n"),
            agent_persona=AgentPersona(name="A", tier="l1", expertise=""),
        ),
    )
    return TicketState(
        run_id=str(uuid4()), parent_run_id=str(uuid4()), run_seed=1,
        company=CompanyProfile(name="A", raw_markdown="# A"),
        current_ticket=ticket,
        current_turn_draft=TurnDraft(
            speaker="agent", content="x", intent="question",
        ),
        verdicts=verdicts or [],
        retry_attempt=retry,
        turn_index=turn_index,
        turns_committed=[
            CommittedTurn(
                id=str(i), ticket_id="t-1", turn_index=i,
                draft=TurnDraft(speaker="agent", content="x", intent="question"),
            )
            for i in range(turns_committed)
        ],
    )


def test_dispatch_turn_checkers_sends_to_three_nodes() -> None:
    state = _state()
    sends = dispatch_turn_checkers(state)
    targets = sorted(s.node for s in sends)
    assert targets == [
        "turn_background_check",
        "turn_consistency_check",
        "turn_scenario_check",
    ]
    for s in sends:
        assert isinstance(s, Send)


def test_aggregate_turn_verdicts_is_noop_passthrough() -> None:
    state = _state(verdicts=[Verdict(checker="x", passed=True)])
    update = aggregate_turn_verdicts(state)
    assert update == {}


def test_route_turn_verdict_all_pass_returns_commit() -> None:
    state = _state(verdicts=[
        Verdict(checker="consistency", passed=True),
        Verdict(checker="background", passed=True),
        Verdict(checker="scenario", passed=True),
    ])
    assert route_turn_verdict(state, max_retries=3) == "commit"


def test_route_turn_verdict_fail_under_budget_returns_regenerate() -> None:
    state = _state(
        verdicts=[
            Verdict(checker="consistency", passed=False, issues=[
                Issue(severity="error", location="x", rule_violated="r",
                      explanation="e"),
            ]),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ],
        retry=1,
    )
    assert route_turn_verdict(state, max_retries=3) == "regenerate"


def test_route_turn_verdict_fail_over_budget_returns_commit_with_warning() -> None:
    state = _state(
        verdicts=[
            Verdict(checker="consistency", passed=False, issues=[
                Issue(severity="error", location="x", rule_violated="r",
                      explanation="e"),
            ]),
            Verdict(checker="background", passed=True),
            Verdict(checker="scenario", passed=True),
        ],
        retry=3,
    )
    assert route_turn_verdict(state, max_retries=3) == "commit_with_warning"


def test_route_turn_loop_more_turns_when_under_max() -> None:
    state = _state(turns_committed=2)
    assert route_turn_loop(state, min_turns=2, max_turns=6) == "more_turns"


def test_route_turn_loop_done_when_at_max() -> None:
    state = _state(turns_committed=6)
    assert route_turn_loop(state, min_turns=2, max_turns=6) == "ticket_done"


def test_route_turn_loop_more_turns_below_min() -> None:
    state = _state(turns_committed=1)
    assert route_turn_loop(state, min_turns=2, max_turns=6) == "more_turns"
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Implement `src/csfd/phases/phase2_cases/routing.py`**

```python
"""Routing helpers for the Phase 2 LangGraph subgraph."""
from __future__ import annotations

from typing import Any

from langgraph.types import Send

from csfd.phases.phase2_cases.state import TicketState


def dispatch_turn_checkers(state: TicketState) -> list[Send]:
    """Fan out the candidate turn to the 3 checkers."""
    payload = {
        "current_turn_draft": state.current_turn_draft,
        "current_ticket": state.current_ticket,
    }
    return [
        Send("turn_consistency_check", payload),
        Send("turn_background_check", payload),
        Send("turn_scenario_check", payload),
    ]


def aggregate_turn_verdicts(state: TicketState) -> dict[str, Any]:
    """No-op fan-in barrier — the Annotated reducer has already accumulated the
    list. Present as a single seam where quorum / weighted-scoring logic could
    be inserted later."""
    return {}


def route_turn_verdict(state: TicketState, *, max_retries: int) -> str:
    """Decide what happens after the 3 turn checkers' verdicts are aggregated."""
    if all(v.passed for v in state.verdicts):
        return "commit"
    if state.retry_attempt >= max_retries:
        return "commit_with_warning"
    return "regenerate"


def route_turn_loop(
    state: TicketState, *, min_turns: int, max_turns: int,
) -> str:
    """Decide whether to generate another turn or close the ticket.

    Below `min_turns`: always continue. At/above `max_turns`: always stop.
    Between: stop (the loop is bounded; Plan 5 may add an LLM-judge here)."""
    n = len(state.turns_committed)
    if n < min_turns:
        return "more_turns"
    if n >= max_turns:
        return "ticket_done"
    return "ticket_done"
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_routing.py -v` → 8 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/routing.py tests/unit/test_phase2_routing.py
git commit -m "feat: add Phase 2 routing: Send dispatch, verdict + turn-loop deciders"
```

---

## Task 10: `build_phase2_graph` subgraph builder

**Files:**
- Create: `src/csfd/phases/phase2_cases/subgraph.py`
- Test: `tests/unit/test_phase2_subgraph.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_phase2_subgraph.py`:
```python
import pytest

from csfd.phases.phase2_cases.subgraph import build_phase2_graph


def test_build_phase2_graph_compiles_with_expected_nodes() -> None:
    graph = build_phase2_graph(
        max_retries=3,
        creative_noise_probability=0.2,
        min_turns=2,
        max_turns=6,
    )
    nodes = graph.get_graph().nodes
    expected = {
        "load_kb",
        "ticket_sampler",
        "ticket_init",
        "turn_writer",
        "creative_noise_gate",
        "creative_noise",
        "turn_check_dispatch",
        "turn_consistency_check",
        "turn_background_check",
        "turn_scenario_check",
        "turn_aggregate",
        "turn_route",
        "turn_loop",
    }
    for n in expected:
        assert n in nodes, f"missing node: {n}"


@pytest.mark.asyncio
async def test_build_phase2_graph_renders_mermaid() -> None:
    graph = build_phase2_graph(
        max_retries=3, creative_noise_probability=0.2,
        min_turns=2, max_turns=6,
    )
    text = graph.get_graph().draw_mermaid()
    assert "turn_writer" in text
    assert "creative_noise" in text
```

- [ ] **Step 2: Confirm failure** — ImportError.

- [ ] **Step 3: Implement `src/csfd/phases/phase2_cases/subgraph.py`**

```python
"""Phase 2 LangGraph subgraph composition."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from csfd.phases.phase2_cases.routing import (
    aggregate_turn_verdicts,
    dispatch_turn_checkers,
    route_turn_loop,
    route_turn_verdict,
)
from csfd.phases.phase2_cases.state import TicketState


def build_phase2_graph(
    *,
    max_retries: int,
    creative_noise_probability: float,  # noqa: ARG001  (used by extended graph in Plan 5)
    min_turns: int,
    max_turns: int,
) -> CompiledStateGraph:
    """Build (and compile) the structural Phase 2 subgraph: load KB, sample
    ticket, init, then iterate turns through writer → creative-noise gate →
    3 parallel checkers → aggregate → route to commit / regenerate /
    commit_with_warning, then loop until ticket_done.

    Like the Phase 1 builder, this is a *structural* subgraph — Plan 5 wires
    the `AgentFactory`, `Database`, and `Phase2Config` into the nodes via
    `functools.partial` and adds the outer multi-ticket loop with the parent
    graph composition."""
    g = StateGraph(TicketState)

    def _placeholder(_state: TicketState) -> dict[str, object]:
        raise NotImplementedError(
            "Bind agent factories + db at subgraph instantiation (Plan 5)."
        )

    for n in (
        "load_kb",
        "ticket_sampler",
        "ticket_init",
        "turn_writer",
        "creative_noise_gate",
        "creative_noise",
        "turn_check_dispatch",
        "turn_consistency_check",
        "turn_background_check",
        "turn_scenario_check",
        "turn_route",
        "turn_loop",
    ):
        g.add_node(n, _placeholder)
    g.add_node("turn_aggregate", aggregate_turn_verdicts)

    g.add_edge(START, "load_kb")
    g.add_edge("load_kb", "ticket_sampler")
    g.add_edge("ticket_sampler", "ticket_init")
    g.add_edge("ticket_init", "turn_writer")

    # Probabilistic gate decides noise vs straight-to-check
    g.add_conditional_edges(
        "turn_writer",
        # Placeholder: real probability comes from cfg at instantiation
        lambda _state: "skip_noise",
        {
            "apply_noise": "creative_noise",
            "skip_noise": "turn_check_dispatch",
        },
    )
    g.add_edge("creative_noise", "turn_check_dispatch")

    g.add_conditional_edges(
        "turn_check_dispatch",
        dispatch_turn_checkers,
        {
            "turn_consistency_check": "turn_consistency_check",
            "turn_background_check": "turn_background_check",
            "turn_scenario_check": "turn_scenario_check",
        },
    )
    g.add_edge("turn_consistency_check", "turn_aggregate")
    g.add_edge("turn_background_check", "turn_aggregate")
    g.add_edge("turn_scenario_check", "turn_aggregate")

    def _route_verdict(state: TicketState) -> str:
        return route_turn_verdict(state, max_retries=max_retries)

    g.add_conditional_edges(
        "turn_aggregate",
        _route_verdict,
        {
            "commit": "turn_loop",
            "commit_with_warning": "turn_loop",
            "regenerate": "turn_writer",
        },
    )

    def _route_loop(state: TicketState) -> str:
        return route_turn_loop(state, min_turns=min_turns, max_turns=max_turns)

    g.add_conditional_edges(
        "turn_loop",
        _route_loop,
        {"more_turns": "turn_writer", "ticket_done": END},
    )

    return g.compile()
```

NOTE: this Plan 4 `build_phase2_graph` is intentionally a *structural* subgraph — Plan 5 will produce a fully-wired graph that injects the `AgentFactory`, `Database`, and `Phase2Config` into the nodes via partials and adds the outer multi-ticket loop with parent-graph composition. The integration test in Task 11 of THIS plan exercises the nodes directly (without the graph) to validate the runtime wiring.

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_phase2_subgraph.py -v` → 2 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/phases/phase2_cases/subgraph.py tests/unit/test_phase2_subgraph.py
git commit -m "feat: add Phase 2 subgraph structural composition"
```

---

## Task 11: End-to-end Phase 2 integration test (no graph; direct node composition)

**Files:**
- Test: `tests/integration/test_phase2_e2e_tiny.py`

Composes the Phase 2 nodes directly (without the LangGraph graph) and demonstrates the full loop on tiny seeds for **both** KB and no-KB paths, asserting:
- `has_kb=False ⇒ L3` hard routing rule
- Noise stratification: with `creative_noise_probability=1.0`, every turn has `noise_applied=True` and a recorded `noise_type`
- Turn count between `min_turns` and `max_turns`

- [ ] **Step 1: Write the test**

`tests/integration/test_phase2_e2e_tiny.py`:
```python
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from csfd.agents.base import Verdict
from csfd.agents.creative_noise import TurnModification
from csfd.agents.factory import AgentFactory
from csfd.models.fake import FakeChatModel
from csfd.phases.phase1_kb.state import (
    CommittedArticle,
    CommittedProblem,
    KBArticleDraft,
    ProblemDraft,
)
from csfd.phases.phase2_cases.nodes import (
    creative_noise_gate,
    creative_noise_node,
    load_kb_node,
    persist_ticket_node,
    persist_turn_node,
    ticket_init_node,
    ticket_sampler_node,
    turn_background_check_node,
    turn_consistency_check_node,
    turn_scenario_check_node,
    turn_writer_node,
)
from csfd.phases.phase2_cases.state import (
    CommittedTurn,
    TicketState,
    TurnDraft,
)
from csfd.prompts.registry import PromptRegistry
from csfd.seeds.company import CompanyProfile
from csfd.settings import (
    AgentLLMConfig,
    Phase2Config,
    TicketsPerProblem,
    TicketTypeWeights,
)
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import (
    KBArticleRecord,
    KBArticleRepo,
    ProblemRecord,
    ProblemRepo,
    RunRecord,
    RunRepo,
    TicketRepo,
    TurnRepo,
)
from csfd.utils.hashing import short_hash


def _seed_phase1_db(tmp_db_path: Path) -> tuple[Database, str, str, str]:
    """Create two problems (1 with KB, 1 without) in a parent phase1 run."""
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    parent_run_id = str(uuid4())
    run_id = str(uuid4())
    for rid, phase, parent in [
        (parent_run_id, "phase1", None), (run_id, "phase2", parent_run_id),
    ]:
        RunRepo(db).create(RunRecord(
            id=rid, phase=phase, parent_run_id=parent, status="running",
            started_at=datetime.now(timezone.utc), completed_at=None,
            run_seed=1, pipeline_version="0.1.0", git_sha=None,
            config_snapshot_json="{}", stats_json=None, error_summary=None,
        ))
    p_kb = str(uuid4())
    p_no = str(uuid4())
    for pid, has_kb in [(p_kb, True), (p_no, False)]:
        ProblemRepo(db).create(ProblemRecord(
            id=pid, run_id=parent_run_id, title=f"P-{pid[:4]}", description="d",
            category="auth", severity="medium", has_kb=has_kb,
            coverage_reasoning="r", coverage_confidence="high",
            metadata_json=None, quality_flag=None, unresolved_issues_json=None,
            created_at=datetime.now(timezone.utc),
        ))
    KBArticleRepo(db).create(KBArticleRecord(
        id=str(uuid4()), run_id=parent_run_id, problem_id=p_kb,
        title="Recover login", content_markdown="## Step 1",
        content_hash=short_hash("## Step 1"),
        troubleshooting_steps_json="[]", prerequisites_json=None,
        metadata_json=None, version=1, quality_flag=None,
        unresolved_issues_json=None, created_at=datetime.now(timezone.utc),
    ))
    return db, run_id, parent_run_id, p_no


def _factory_for_e2e(
    canned_turn: TurnDraft,
    canned_verdict: Verdict,
    canned_noise: TurnModification,
) -> AgentFactory:
    reg = PromptRegistry(root=Path("prompts"))
    reg.load()
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5",
                         temperature=0.7)
    return AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(structured={
            TurnDraft: canned_turn,
            Verdict: canned_verdict,
            TurnModification: canned_noise,
        }),
        agent_configs={
            "turn_writer": cfg,
            "turn_consistency": cfg,
            "turn_background": cfg,
            "turn_scenario": cfg,
            "creative_noise": cfg,
        },
    )


def _phase2_cfg(noise_prob: float) -> Phase2Config:
    return Phase2Config(
        tickets_per_problem=TicketsPerProblem(has_kb=(1, 1), no_kb=(1, 1)),
        ticket_type_weights=TicketTypeWeights(
            has_kb={"docs_request": 0.0, "l1": 1.0, "l2": 0.0, "l3": 0.0},
            no_kb={"l3": 1.0},
        ),
        creative_noise_probability=noise_prob,
        noise_type_weights={"typos_informal_phrasing": 1.0},
        min_turns_per_ticket=2,
        max_turns_per_ticket=2,
    )


async def _run_ticket(
    state: TicketState,
    *,
    factory: AgentFactory,
    db: Database,
    cfg: Phase2Config,
) -> TicketState:
    """Run the inner per-ticket loop until ticket_done."""
    while len(state.turns_committed) < cfg.max_turns_per_ticket:
        # 1. Write a turn
        update = await turn_writer_node(state, factory=factory)
        state = state.model_copy(update=update)
        # 2. Creative noise gate
        gate = creative_noise_gate(state, probability=cfg.creative_noise_probability)
        if gate == "apply_noise":
            update = await creative_noise_node(
                state, factory=factory,
                noise_type_weights=cfg.noise_type_weights,
            )
            state = state.model_copy(update=update)
        # 3. Run 3 checkers in parallel
        payload = {"current_turn_draft": state.current_turn_draft}
        verdicts = await asyncio.gather(
            turn_consistency_check_node(payload, factory=factory),
            turn_background_check_node(payload, factory=factory),
            turn_scenario_check_node(payload, factory=factory),
        )
        state = state.model_copy(update={
            "verdicts": [v["verdicts"][0] for v in verdicts],
        })
        # 4. All-pass commit (canned verdicts pass in this test)
        assert all(v.passed for v in state.verdicts)
        # 5. Persist turn + reset
        assert state.current_turn_draft is not None
        assert state.current_ticket is not None
        committed = CommittedTurn(
            id=str(uuid4()),
            ticket_id=state.current_ticket.id,
            turn_index=state.turn_index,
            draft=state.current_turn_draft,
        )
        persist_turn_node(state, db=db, turn=committed)
        state = state.model_copy(update={
            "turns_committed": state.turns_committed + [committed],
            "current_turn_draft": None,
            "noise_applied": False,
            "noise_type": None,
            "turn_index": state.turn_index + 1,
            "verdicts": [],
        })
    return state


@pytest.mark.asyncio
async def test_phase2_no_kb_forces_l3_and_noise_always_applied(
    tmp_db_path: Path,
) -> None:
    db, run_id, parent_run_id, p_no_id = _seed_phase1_db(tmp_db_path)
    canned_turn = TurnDraft(
        speaker="agent", content="Diagnosing now.", intent="clarification",
    )
    canned_verdict = Verdict(checker="x", passed=True)
    canned_noise = TurnModification(
        modified_content="diagnosing... brb",
        noise_type="typos_informal_phrasing",
        rationale="casual",
    )
    factory = _factory_for_e2e(canned_turn, canned_verdict, canned_noise)
    cfg = _phase2_cfg(noise_prob=1.0)

    # Initial state — load KB first, then filter pool to no-KB problem so the
    # sampler picks it.
    state = TicketState(
        run_id=run_id, parent_run_id=parent_run_id, run_seed=1,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
    )
    state = state.model_copy(update=await load_kb_node(state, db=db))
    state = state.model_copy(update={
        "problem_pool": [p for p in state.problem_pool if p.id == p_no_id],
    })

    # 1. Sample (problem, ticket_type)
    update = await ticket_sampler_node(state, cfg=cfg)
    assert update["ticket_type"] == "l3"  # hard rule

    # 2. Init ticket
    problem = state.problem_pool[0]
    update = await ticket_init_node(
        state, problem=problem, ticket_type="l3", kb_article_id=None,
    )
    state = state.model_copy(update=update)
    assert state.current_ticket is not None
    assert state.current_ticket.draft.ticket_type == "l3"

    # 3. Persist ticket
    persist_ticket_node(state, db=db, ticket=state.current_ticket)

    # 4. Inner turn loop
    state = await _run_ticket(state, factory=factory, db=db, cfg=cfg)

    # Assertions
    persisted_ticket = TicketRepo(db)
    # query by SELECT
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT ticket_type FROM tickets WHERE run_id = ?", (run_id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["ticket_type"] == "l3"

    turn_rows = TurnRepo(db).list_for_ticket(state.current_ticket.id)
    assert len(turn_rows) == 2  # max_turns_per_ticket
    for t in turn_rows:
        assert t.noise_applied is True
        assert t.noise_type == "typos_informal_phrasing"


@pytest.mark.asyncio
async def test_phase2_with_kb_uses_l1_and_no_noise_when_probability_zero(
    tmp_db_path: Path,
) -> None:
    db, run_id, parent_run_id, _p_no_id = _seed_phase1_db(tmp_db_path)
    canned_turn = TurnDraft(
        speaker="agent",
        content="Hi, let me help via our reset link.",
        intent="resolution",
    )
    canned_verdict = Verdict(checker="x", passed=True)
    canned_noise = TurnModification(
        modified_content="unused", noise_type="x", rationale="",
    )
    factory = _factory_for_e2e(canned_turn, canned_verdict, canned_noise)
    cfg = _phase2_cfg(noise_prob=0.0)

    state = TicketState(
        run_id=run_id, parent_run_id=parent_run_id, run_seed=2,
        company=CompanyProfile(name="Acme", raw_markdown="# Acme"),
    )
    state = state.model_copy(update=await load_kb_node(state, db=db))
    state = state.model_copy(update={
        "problem_pool": [p for p in state.problem_pool if p.has_kb],
    })

    update = await ticket_sampler_node(state, cfg=cfg)
    assert update["ticket_type"] == "l1"

    problem = state.problem_pool[0]
    kb_article = state.kb_by_problem[problem.id]
    update = await ticket_init_node(
        state, problem=problem, ticket_type="l1", kb_article_id=kb_article.id,
    )
    state = state.model_copy(update=update)
    persist_ticket_node(state, db=db, ticket=state.current_ticket)
    state = await _run_ticket(state, factory=factory, db=db, cfg=cfg)

    turn_rows = TurnRepo(db).list_for_ticket(state.current_ticket.id)
    assert len(turn_rows) == 2
    for t in turn_rows:
        assert t.noise_applied is False
        assert t.noise_type is None
```

- [ ] **Step 2: Verify** — `uv run pytest tests/integration/test_phase2_e2e_tiny.py -v` → 2 pass; `uv run pytest -q` no regressions; mypy + ruff clean.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase2_e2e_tiny.py
git commit -m "test: end-to-end Phase 2 pipeline with FakeChatModel (KB + no-KB paths)"
```

---

## Task 12: `phases/phase2_cases/__init__.py` re-exports + full-suite verification

**Files:**
- Modify: `src/csfd/phases/phase2_cases/__init__.py`

- [ ] **Step 1: Write `src/csfd/phases/phase2_cases/__init__.py`**

```python
"""csfd.phases.phase2_cases — Phase 2 case-generation subgraph."""
from csfd.phases.phase2_cases.nodes import (
    creative_noise_gate,
    creative_noise_node,
    load_kb_node,
    persist_ticket_node,
    persist_turn_node,
    pick_ticket_count,
    ticket_init_node,
    ticket_sampler_node,
    turn_background_check_node,
    turn_consistency_check_node,
    turn_scenario_check_node,
    turn_writer_node,
)
from csfd.phases.phase2_cases.routing import (
    aggregate_turn_verdicts,
    dispatch_turn_checkers,
    route_turn_loop,
    route_turn_verdict,
)
from csfd.phases.phase2_cases.sampler import (
    sample_ticket_count,
    sample_ticket_type,
)
from csfd.phases.phase2_cases.state import (
    AgentPersona,
    CommittedTicket,
    CommittedTurn,
    CustomerPersona,
    Phase2Stats,
    TicketDraft,
    TicketState,
    TurnDraft,
)
from csfd.phases.phase2_cases.subgraph import build_phase2_graph

__all__ = [
    "AgentPersona",
    "CommittedTicket",
    "CommittedTurn",
    "CustomerPersona",
    "Phase2Stats",
    "TicketDraft",
    "TicketState",
    "TurnDraft",
    "aggregate_turn_verdicts",
    "build_phase2_graph",
    "creative_noise_gate",
    "creative_noise_node",
    "dispatch_turn_checkers",
    "load_kb_node",
    "persist_ticket_node",
    "persist_turn_node",
    "pick_ticket_count",
    "route_turn_loop",
    "route_turn_verdict",
    "sample_ticket_count",
    "sample_ticket_type",
    "ticket_init_node",
    "ticket_sampler_node",
    "turn_background_check_node",
    "turn_consistency_check_node",
    "turn_scenario_check_node",
    "turn_writer_node",
]
```

- [ ] **Step 2: Verify** — `uv run python -c "from csfd.phases.phase2_cases import TicketState, build_phase2_graph; print('ok')"` → prints `ok`.

- [ ] **Step 3: Full-suite check**
  - `uv run pytest -v` — all green (Plan 3 tests + new Plan 4 tests)
  - `uv run mypy src tests` — Success, no issues
  - `uv run ruff check src tests` + `uv run ruff format --check src tests` — clean
  - If ruff format made changes, commit `chore: ruff format after Plan 4 work`

- [ ] **Step 4: Commit**

```bash
git add src/csfd/phases/phase2_cases/__init__.py
git commit -m "feat: re-export Phase 2 symbols at csfd.phases.phase2_cases"
```

- [ ] **Step 5: Capture summary** — `git log --oneline | head -25`, total test count, source file count.

## Plan 4 completion criteria

- [ ] All 12 tasks complete; tests pass.
- [ ] `csfd.phases.phase2_cases` package exposes the listed symbols.
- [ ] End-to-end test demonstrates both paths: with-KB → L1 + no noise (probability=0), no-KB → L3 + noise on every turn (probability=1.0); turns persist to SQLite with correct `noise_applied` / `noise_type`.
- [ ] `build_phase2_graph` compiles structurally and renders mermaid.
- [ ] Full suite green; mypy strict clean; ruff clean.

Plan 4 done. Hand off to **Plan 5 — Integration & Polish**.
