# CSFD Agent Framework Implementation Plan (Plan 2 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Build the 5-role agent framework on top of the Plan 1 foundation. After Plan 2: typed `AgentRole` ABC + `Verdict`/`Issue` schemas + concrete `Generator` / `Checker` / `CreativeNoise` classes + an `AgentFactory` that wires the prompt registry, LLM provider registry, and per-agent settings into ready-to-invoke instances. An end-to-end test demonstrates a Generator producing a typed artifact, three Checkers producing Verdicts, the verdict-aggregation routing back to the Generator with structured `Issue` feedback, and a retry succeeding — all using `FakeChatModel`.

**Architecture:**
- Every agent is a thin wrapper around `(prompt_handle, output_schema, llm_client)` with a single async `invoke(ctx) -> BaseModel`.
- Generators emit their artifact schema; Checkers emit `Verdict`; CreativeNoise emits a turn modification record.
- LLM responses are typed via LangChain's `with_structured_output(schema, method="json_schema")`. Schema-validation failures raise `SchemaValidationError` (defined in Plan 1) — the graph layer (Plans 3-4) will convert this into a synthetic failed `Verdict` for the next retry.
- `AgentContext` carries `inputs` + `prior_committed_artifacts` + `retry_attempt` + `prior_verdicts` — the structured feedback that drives the evaluator-optimizer loop.
- `FakeChatModel` is enhanced to support `with_structured_output(schema)`, returning canned Pydantic instances.
- Each invocation can write an `AgentTraceRecord` row via a small helper.

**Tech Stack:** Python 3.12+, Pydantic v2, LangChain core/anthropic/openai, Jinja2, structlog. Builds on Plan 1.

**Working directory:** `/Users/joao.augusto/Documents/Customer Service Fake Data`

---

## File Structure (Plan 2 scope)

| Path | Responsibility |
|---|---|
| `src/csfd/agents/base.py` | `Issue`, `Verdict`, `AgentContext`, `AgentRole` ABC |
| `src/csfd/agents/generator.py` | `Generator(AgentRole)` |
| `src/csfd/agents/checker.py` | `Checker(AgentRole)` — always emits `Verdict` |
| `src/csfd/agents/creative_noise.py` | `CreativeNoise(AgentRole)` — emits `TurnModification` |
| `src/csfd/agents/factory.py` | `AgentFactory` composing PromptRegistry + build_llm + AgentLLMConfig |
| `src/csfd/agents/tracing.py` | `record_agent_trace(...)` helper writing to `AgentTraceRepo` |
| `src/csfd/agents/__init__.py` | re-exports |
| `src/csfd/models/fake.py` | (modify) add `with_structured_output` support |
| `prompts/phase1/*_test_stub.md.j2`, `prompts/phase2/*_test_stub.md.j2` | minimal Jinja stubs that include `prior_verdicts` rendering — real bodies land in Plans 3-4 |
| `tests/unit/test_agent_base.py` | Issue/Verdict/AgentContext shape tests |
| `tests/unit/test_agent_generator.py` | Generator with FakeChatModel returns typed artifact |
| `tests/unit/test_agent_checker.py` | Checker returns Verdict |
| `tests/unit/test_agent_creative_noise.py` | CreativeNoise returns modification |
| `tests/unit/test_agent_factory.py` | factory builds agents from settings |
| `tests/unit/test_agent_tracing.py` | tracing helper writes a row |
| `tests/integration/test_feedback_loop.py` | full Generator→Checker→retry-with-issues cycle with FakeChatModel |

---

## Task 1: Issue + Verdict + AgentContext schemas

**Files:**
- Create: `src/csfd/agents/base.py`
- Test: `tests/unit/test_agent_base.py`

- [ ] **Step 1: Write failing test**

`tests/unit/test_agent_base.py`:
```python
import pytest
from pydantic import ValidationError

from csfd.agents.base import AgentContext, Issue, Verdict


def test_issue_requires_severity_and_location() -> None:
    issue = Issue(
        severity="error",
        location="turn[2].assistant_message",
        rule_violated="L1_must_not_mention_internal_tools",
        explanation="Mentioned Jira ticket id",
        suggested_fix="Remove or replace",
    )
    assert issue.severity == "error"
    assert issue.suggested_fix == "Remove or replace"


def test_issue_severity_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        Issue(
            severity="critical",  # type: ignore[arg-type]
            location="x",
            rule_violated="r",
            explanation="e",
        )


def test_verdict_pass_with_no_issues() -> None:
    v = Verdict(checker="consistency", passed=True)
    assert v.passed is True
    assert v.issues == []


def test_verdict_fail_carries_issues() -> None:
    v = Verdict(
        checker="background",
        passed=False,
        issues=[Issue(
            severity="warning", location="problem.description",
            rule_violated="tone_company_voice",
            explanation="Too informal",
        )],
    )
    assert v.passed is False
    assert len(v.issues) == 1


def test_agent_context_defaults() -> None:
    ctx = AgentContext(inputs={"a": 1})
    assert ctx.inputs == {"a": 1}
    assert ctx.prior_committed == {}
    assert ctx.retry_attempt == 0
    assert ctx.prior_verdicts == []
```

- [ ] **Step 2: Confirm failure** — `uv run pytest tests/unit/test_agent_base.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `src/csfd/agents/base.py`**

```python
"""Base agent types: Issue, Verdict, AgentContext, AgentRole."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class Issue(BaseModel):
    """Structured defect raised by a checker."""

    severity: Literal["error", "warning"]
    location: str
    rule_violated: str
    explanation: str
    suggested_fix: str | None = None


class Verdict(BaseModel):
    """A checker's judgement on a candidate artifact."""

    checker: str
    passed: bool
    issues: list[Issue] = Field(default_factory=list)


class AgentContext(BaseModel):
    """Inputs passed to every agent invocation."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    prior_committed: dict[str, Any] = Field(default_factory=dict)
    retry_attempt: int = 0
    prior_verdicts: list[Verdict] = Field(default_factory=list)


class AgentRole(ABC):
    """ABC for all agent roles. Subclasses implement `invoke`."""

    name: str

    @abstractmethod
    async def invoke(self, ctx: AgentContext) -> BaseModel:
        """Run the agent against the context and return its typed output."""
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_agent_base.py -v` → 5 pass; `uv run mypy src/csfd/agents/base.py tests/unit/test_agent_base.py` → clean; `uv run ruff check ...` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/agents/base.py tests/unit/test_agent_base.py
git commit -m "feat: add Issue, Verdict, AgentContext schemas and AgentRole ABC"
```

---

## Task 2: Enhance `FakeChatModel` to support `with_structured_output`

**Files:**
- Modify: `src/csfd/models/fake.py`
- Test: `tests/unit/test_fake_structured.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_fake_structured.py`:
```python
from pydantic import BaseModel

from csfd.models.fake import FakeChatModel


class _Out(BaseModel):
    x: int
    label: str


def test_fake_with_structured_output_returns_typed_instance() -> None:
    canned = _Out(x=42, label="hi")
    llm = FakeChatModel(structured={_Out: canned})
    runnable = llm.with_structured_output(_Out)
    result = runnable.invoke("anything")
    assert isinstance(result, _Out)
    assert result.x == 42
    assert result.label == "hi"


def test_fake_with_structured_output_raises_when_no_canned() -> None:
    llm = FakeChatModel()
    runnable = llm.with_structured_output(_Out)
    import pytest
    with pytest.raises(KeyError):
        runnable.invoke("anything")
```

- [ ] **Step 2: Confirm failure** — `uv run pytest tests/unit/test_fake_structured.py -v` → AttributeError (no `structured` parameter) or fails the assertion.

- [ ] **Step 3: Modify `src/csfd/models/fake.py`**

Add a `structured: dict[type[BaseModel], BaseModel]` field, and override `with_structured_output` to return a `Runnable` that returns the canned instance:

```python
"""In-process fake chat model for unit and integration tests."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field


class FakeChatModel(BaseChatModel):
    canned: dict[str, str] = Field(default_factory=dict)
    structured: dict[type[BaseModel], BaseModel] = Field(default_factory=dict)
    call_log: list[Any] = Field(default_factory=list)

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

    def with_structured_output(  # type: ignore[override]
        self,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[Any, BaseModel]:
        canned = self.structured

        def _return_canned(_inputs: Any) -> BaseModel:
            if schema not in canned:
                raise KeyError(
                    f"FakeChatModel has no canned structured output for {schema.__name__}"
                )
            return canned[schema]

        return RunnableLambda(_return_canned)
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_fake_structured.py tests/unit/test_models_registry.py -v` → all pass (no regression on the existing FakeChatModel tests); mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/models/fake.py tests/unit/test_fake_structured.py
git commit -m "feat: add structured-output support to FakeChatModel"
```

---

## Task 3: `Generator` class

**Files:**
- Create: `src/csfd/agents/generator.py`
- Test: `tests/unit/test_agent_generator.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_agent_generator.py`:
```python
import jinja2
import pytest
from pydantic import BaseModel

from csfd.agents.base import AgentContext, Issue, Verdict
from csfd.agents.generator import Generator
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle


class _Artifact(BaseModel):
    summary: str


def _handle(source: str = "Generate something. {{ inputs.x | default('') }} {% for v in prior_verdicts %}{% for i in v.issues %}- {{ i.explanation }}{% endfor %}{% endfor %}") -> PromptHandle:
    env = jinja2.Environment(autoescape=False)
    return PromptHandle(name="t", template=env.from_string(source), source=source, version="abc123abc123")


@pytest.mark.asyncio
async def test_generator_returns_typed_artifact() -> None:
    canned = _Artifact(summary="ok")
    llm = FakeChatModel(structured={_Artifact: canned})
    gen = Generator(name="g", prompt=_handle(), output_schema=_Artifact, llm=llm)
    out = await gen.invoke(AgentContext(inputs={"x": "hello"}))
    assert isinstance(out, _Artifact)
    assert out.summary == "ok"


@pytest.mark.asyncio
async def test_generator_renders_prior_verdicts_into_prompt() -> None:
    canned = _Artifact(summary="ok2")
    llm = FakeChatModel(structured={_Artifact: canned})
    gen = Generator(name="g", prompt=_handle(), output_schema=_Artifact, llm=llm)
    verdict = Verdict(
        checker="consistency", passed=False,
        issues=[Issue(severity="error", location="x", rule_violated="r", explanation="Internal id leaked")],
    )
    await gen.invoke(AgentContext(inputs={"x": "hello"}, prior_verdicts=[verdict]))
    rendered = gen.last_rendered_prompt or ""
    assert "Internal id leaked" in rendered
```

- [ ] **Step 2: Confirm failure** — `uv run pytest tests/unit/test_agent_generator.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `src/csfd/agents/generator.py`**

```python
"""Generator agent — produces a structured artifact from a prompt + context."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from csfd.agents.base import AgentContext, AgentRole
from csfd.prompts.registry import PromptHandle


class Generator(AgentRole):
    """Agent role that emits an artifact of a given Pydantic schema."""

    def __init__(
        self,
        *,
        name: str,
        prompt: PromptHandle,
        output_schema: type[BaseModel],
        llm: BaseChatModel,
    ) -> None:
        self.name = name
        self.prompt = prompt
        self.output_schema = output_schema
        self.llm = llm
        self.last_rendered_prompt: str | None = None

    async def invoke(self, ctx: AgentContext) -> BaseModel:
        rendered = self.prompt.template.render(
            inputs=ctx.inputs,
            prior_committed=ctx.prior_committed,
            retry_attempt=ctx.retry_attempt,
            prior_verdicts=[v.model_dump() for v in ctx.prior_verdicts],
        )
        self.last_rendered_prompt = rendered
        runnable = self.llm.with_structured_output(self.output_schema)
        result = await runnable.ainvoke(rendered)
        if not isinstance(result, BaseModel):
            raise TypeError(
                f"Expected {self.output_schema.__name__}, got {type(result).__name__}"
            )
        return result
```

- [ ] **Step 4: Verify** — `uv run pytest tests/unit/test_agent_generator.py -v` → 2 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/agents/generator.py tests/unit/test_agent_generator.py
git commit -m "feat: add Generator agent emitting typed artifacts"
```

Note for the test: `prior_verdicts` is rendered as a list of dicts. The Jinja template iterates `prior_verdicts` (a list of dicts) and accesses `.issues` (a list of dicts). The test fixture template uses `{% for v in prior_verdicts %}{% for i in v.issues %}- {{ i.explanation }}{% endfor %}{% endfor %}` — this works because Jinja access dict keys as attributes.

---

## Task 4: `Checker` class

**Files:**
- Create: `src/csfd/agents/checker.py`
- Test: `tests/unit/test_agent_checker.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_agent_checker.py`:
```python
import jinja2
import pytest

from csfd.agents.base import AgentContext, Verdict
from csfd.agents.checker import Checker
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle


def _handle() -> PromptHandle:
    src = "Inspect: {{ inputs.artifact }}"
    env = jinja2.Environment(autoescape=False)
    return PromptHandle(name="c", template=env.from_string(src), source=src, version="cccccccccccc")


@pytest.mark.asyncio
async def test_checker_emits_passing_verdict() -> None:
    canned = Verdict(checker="consistency", passed=True)
    llm = FakeChatModel(structured={Verdict: canned})
    checker = Checker(name="consistency", prompt=_handle(), llm=llm)
    out = await checker.invoke(AgentContext(inputs={"artifact": "x"}))
    assert isinstance(out, Verdict)
    assert out.passed is True


@pytest.mark.asyncio
async def test_checker_overrides_checker_name_to_match_instance() -> None:
    canned = Verdict(checker="WRONG", passed=False)
    llm = FakeChatModel(structured={Verdict: canned})
    checker = Checker(name="background", prompt=_handle(), llm=llm)
    out = await checker.invoke(AgentContext(inputs={"artifact": "x"}))
    assert isinstance(out, Verdict)
    assert out.checker == "background"
```

- [ ] **Step 2: Confirm failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `src/csfd/agents/checker.py`**

```python
"""Checker agent — emits a Verdict on a candidate artifact."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from csfd.agents.base import AgentContext, AgentRole, Verdict
from csfd.agents.generator import Generator
from csfd.prompts.registry import PromptHandle


class Checker(AgentRole):
    """Agent role that always emits a Verdict. Internally delegates to a Generator
    bound to the Verdict schema, then forces the verdict's `checker` field to match
    this checker's own name (the LLM is allowed to fill it but we own the identity).
    """

    def __init__(
        self,
        *,
        name: str,
        prompt: PromptHandle,
        llm: BaseChatModel,
    ) -> None:
        self.name = name
        self._inner = Generator(name=name, prompt=prompt, output_schema=Verdict, llm=llm)

    async def invoke(self, ctx: AgentContext) -> Verdict:
        out: BaseModel = await self._inner.invoke(ctx)
        if not isinstance(out, Verdict):
            raise TypeError(f"Checker expected Verdict, got {type(out).__name__}")
        # Force checker identity to match this instance regardless of what the LLM emitted.
        return out.model_copy(update={"checker": self.name})
```

- [ ] **Step 4: Verify** — pytest 2 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/agents/checker.py tests/unit/test_agent_checker.py
git commit -m "feat: add Checker agent emitting verdicts with enforced identity"
```

---

## Task 5: `CreativeNoise` class

**Files:**
- Create: `src/csfd/agents/creative_noise.py`
- Test: `tests/unit/test_agent_creative_noise.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_agent_creative_noise.py`:
```python
import jinja2
import pytest
from pydantic import BaseModel

from csfd.agents.base import AgentContext
from csfd.agents.creative_noise import CreativeNoise, TurnModification
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle


def _handle() -> PromptHandle:
    src = "Apply noise: {{ inputs.turn_content }}"
    env = jinja2.Environment(autoescape=False)
    return PromptHandle(name="n", template=env.from_string(src), source=src, version="nnnnnnnnnnnn")


@pytest.mark.asyncio
async def test_creative_noise_returns_modified_turn() -> None:
    canned = TurnModification(
        modified_content="hey help my printer is on fire :(",
        noise_type="typos_informal_phrasing",
        rationale="Customer is panicking",
    )
    llm = FakeChatModel(structured={TurnModification: canned})
    noise = CreativeNoise(name="creative_noise", prompt=_handle(), llm=llm)
    out = await noise.invoke(AgentContext(inputs={"turn_content": "help my printer is on fire"}))
    assert isinstance(out, TurnModification)
    assert out.noise_type == "typos_informal_phrasing"
    assert "fire" in out.modified_content
```

- [ ] **Step 2: Confirm failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `src/csfd/agents/creative_noise.py`**

```python
"""CreativeNoise agent — probabilistically injects realistic complexity into a turn."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from csfd.agents.base import AgentContext, AgentRole
from csfd.agents.generator import Generator
from csfd.prompts.registry import PromptHandle


class TurnModification(BaseModel):
    modified_content: str
    noise_type: str
    rationale: str = Field(default="")


class CreativeNoise(AgentRole):
    """Agent role that returns a modified turn with a noise label."""

    def __init__(
        self,
        *,
        name: str,
        prompt: PromptHandle,
        llm: BaseChatModel,
    ) -> None:
        self.name = name
        self._inner = Generator(
            name=name, prompt=prompt, output_schema=TurnModification, llm=llm
        )

    async def invoke(self, ctx: AgentContext) -> TurnModification:
        out: BaseModel = await self._inner.invoke(ctx)
        if not isinstance(out, TurnModification):
            raise TypeError(f"Expected TurnModification, got {type(out).__name__}")
        return out
```

- [ ] **Step 4: Verify** — pytest 1 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/agents/creative_noise.py tests/unit/test_agent_creative_noise.py
git commit -m "feat: add CreativeNoise agent emitting TurnModification"
```

---

## Task 6: `AgentFactory`

**Files:**
- Create: `src/csfd/agents/factory.py`
- Test: `tests/unit/test_agent_factory.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_agent_factory.py`:
```python
from pathlib import Path

import jinja2

from csfd.agents.factory import AgentFactory
from csfd.agents.generator import Generator
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle, PromptRegistry
from csfd.settings import AgentLLMConfig


def _registry(tmp_path: Path) -> PromptRegistry:
    (tmp_path / "phase1").mkdir()
    (tmp_path / "phase1" / "problem_generator.md.j2").write_text("Hi {{ inputs.x }}")
    reg = PromptRegistry(root=tmp_path)
    reg.load()
    return reg


def test_factory_builds_generator_with_bound_llm_and_prompt(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.5)
    factory = AgentFactory(
        prompts=reg,
        llm_builder=lambda c: FakeChatModel(canned={"default": "stub"}),
        agent_configs={"problem_brainstorm": cfg},
    )
    gen = factory.build_generator(
        name="problem_brainstorm",
        prompt_name="phase1.problem_generator",
        output_schema_factory=lambda: __import__("pydantic").BaseModel,
    )
    assert isinstance(gen, Generator)
    assert gen.name == "problem_brainstorm"
    assert isinstance(gen.prompt, PromptHandle)
    assert gen.prompt.name == "phase1.problem_generator"
```

- [ ] **Step 2: Confirm failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `src/csfd/agents/factory.py`**

```python
"""AgentFactory composes prompt registry + LLM builder + per-agent config into agents."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from csfd.agents.checker import Checker
from csfd.agents.creative_noise import CreativeNoise
from csfd.agents.generator import Generator
from csfd.prompts.registry import PromptRegistry
from csfd.settings import AgentLLMConfig


LLMBuilder = Callable[[AgentLLMConfig], BaseChatModel]


@dataclass(slots=True)
class AgentFactory:
    prompts: PromptRegistry
    llm_builder: LLMBuilder
    agent_configs: dict[str, AgentLLMConfig]

    def _llm_for(self, name: str) -> BaseChatModel:
        if name not in self.agent_configs:
            raise KeyError(f"No agent config for: {name}")
        return self.llm_builder(self.agent_configs[name])

    def build_generator(
        self,
        *,
        name: str,
        prompt_name: str,
        output_schema_factory: Callable[[], type[BaseModel]],
    ) -> Generator:
        return Generator(
            name=name,
            prompt=self.prompts.get(prompt_name),
            output_schema=output_schema_factory(),
            llm=self._llm_for(name),
        )

    def build_checker(self, *, name: str, prompt_name: str) -> Checker:
        return Checker(
            name=name,
            prompt=self.prompts.get(prompt_name),
            llm=self._llm_for(name),
        )

    def build_creative_noise(self, *, name: str, prompt_name: str) -> CreativeNoise:
        return CreativeNoise(
            name=name,
            prompt=self.prompts.get(prompt_name),
            llm=self._llm_for(name),
        )
```

- [ ] **Step 4: Verify** — pytest 1 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/agents/factory.py tests/unit/test_agent_factory.py
git commit -m "feat: add AgentFactory composing prompts, LLM, and agent configs"
```

---

## Task 7: `record_agent_trace` helper

**Files:**
- Create: `src/csfd/agents/tracing.py`
- Test: `tests/unit/test_agent_tracing.py`

- [ ] **Step 1: Failing test**

`tests/unit/test_agent_tracing.py`:
```python
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from csfd.agents.tracing import record_agent_trace
from csfd.storage.db import Database
from csfd.storage.migrations.runner import apply_migrations
from csfd.storage.repository import AgentTraceRepo, RunRecord, RunRepo


def test_record_agent_trace_writes_row(tmp_db_path: Path) -> None:
    db = Database(path=tmp_db_path)
    apply_migrations(db)
    run_id = str(uuid4())
    RunRepo(db).create(RunRecord(
        id=run_id, phase="phase1", parent_run_id=None, status="running",
        started_at=datetime.now(timezone.utc), completed_at=None, run_seed=1,
        pipeline_version="0.1.0", git_sha=None, config_snapshot_json="{}",
        stats_json=None, error_summary=None,
    ))
    trace_id = record_agent_trace(
        repo=AgentTraceRepo(db),
        run_id=run_id, thread_id=run_id,
        node_name="problem_brainstorm", agent_role="generator",
        artifact_type="problem", artifact_id=None, attempt=0,
        prompt_id="abc123abc123",
        input_obj={"x": 1}, output_obj={"y": 2},
        verdict=None, verdict_issues=None,
        model_provider="fake", model_id="fake-v1",
        tokens_in=10, tokens_out=20, cost_usd=0.001,
        latency_ms=42, parent_trace_id=None,
        status="ok", error_class=None, error_message=None,
    )
    assert trace_id
    rows = AgentTraceRepo(db).list_for_run(run_id)
    assert len(rows) == 1
    assert rows[0].agent_role == "generator"
    assert json.loads(rows[0].input_json)["x"] == 1
```

- [ ] **Step 2: Confirm failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `src/csfd/agents/tracing.py`**

```python
"""Helper to persist a single agent invocation as an AgentTraceRecord."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from csfd.storage.repository import AgentTraceRecord, AgentTraceRepo


def record_agent_trace(
    *,
    repo: AgentTraceRepo,
    run_id: str,
    thread_id: str,
    node_name: str,
    agent_role: str,
    artifact_type: str,
    artifact_id: str | None,
    attempt: int,
    prompt_id: str,
    input_obj: Any,
    output_obj: Any,
    verdict: str | None,
    verdict_issues: Any,
    model_provider: str,
    model_id: str,
    tokens_in: int | None,
    tokens_out: int | None,
    cost_usd: float | None,
    latency_ms: int | None,
    parent_trace_id: str | None,
    status: str,
    error_class: str | None,
    error_message: str | None,
) -> str:
    trace_id = str(uuid4())
    repo.create(AgentTraceRecord(
        id=trace_id,
        run_id=run_id,
        thread_id=thread_id,
        node_name=node_name,
        agent_role=agent_role,
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        attempt=attempt,
        prompt_id=prompt_id,
        input_json=json.dumps(input_obj, default=str),
        output_json=json.dumps(output_obj, default=str) if output_obj is not None else None,
        verdict=verdict,
        verdict_issues_json=json.dumps(verdict_issues, default=str) if verdict_issues is not None else None,
        model_provider=model_provider,
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd_estimated=cost_usd,
        latency_ms=latency_ms,
        parent_trace_id=parent_trace_id,
        status=status,
        error_class=error_class,
        error_message=error_message,
        created_at=datetime.now(timezone.utc),
    ))
    return trace_id
```

- [ ] **Step 4: Verify** — pytest 1 pass; mypy + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/agents/tracing.py tests/unit/test_agent_tracing.py
git commit -m "feat: add record_agent_trace helper for per-invocation logging"
```

---

## Task 8: `agents/__init__.py` re-exports

**Files:**
- Modify: `src/csfd/agents/__init__.py`
- Test: extend `tests/unit/test_smoke_imports.py` (existing) to also check the new symbols

- [ ] **Step 1: Modify `src/csfd/agents/__init__.py`**

```python
"""csfd.agents — agent role framework."""
from csfd.agents.base import AgentContext, AgentRole, Issue, Verdict
from csfd.agents.checker import Checker
from csfd.agents.creative_noise import CreativeNoise, TurnModification
from csfd.agents.factory import AgentFactory
from csfd.agents.generator import Generator
from csfd.agents.tracing import record_agent_trace

__all__ = [
    "AgentContext",
    "AgentFactory",
    "AgentRole",
    "Checker",
    "CreativeNoise",
    "Generator",
    "Issue",
    "TurnModification",
    "Verdict",
    "record_agent_trace",
]
```

- [ ] **Step 2: Append to `tests/unit/test_smoke_imports.py`**

```python
def test_agents_package_reexports() -> None:
    import csfd.agents as a
    assert all(hasattr(a, sym) for sym in [
        "AgentContext", "AgentRole", "AgentFactory",
        "Issue", "Verdict",
        "Generator", "Checker", "CreativeNoise",
        "TurnModification", "record_agent_trace",
    ])
```

- [ ] **Step 3: Verify** — `uv run pytest tests/unit/test_smoke_imports.py -v` → all pass; mypy + ruff clean.

- [ ] **Step 4: Commit**

```bash
git add src/csfd/agents/__init__.py tests/unit/test_smoke_imports.py
git commit -m "feat: expose agent framework symbols at csfd.agents"
```

---

## Task 9: Placeholder prompt templates for Phase 1 + Phase 2

**Files:**
- Create: `prompts/phase1/problem_generator.md.j2`, `prompts/phase1/article_consistency.md.j2`, `prompts/phase2/turn_generator.md.j2`, `prompts/phase2/turn_consistency.md.j2`, `prompts/phase2/creative_noise.md.j2`

These are STUBS — real bodies land in Plans 3 and 4. The purpose is to provide consumable templates that exercise the feedback-loop rendering.

- [ ] **Step 1: `prompts/phase1/problem_generator.md.j2`**

```jinja
You are brainstorming candidate customer problems for {{ inputs.company_name | default('the company') }}.

Inputs:
{{ inputs | tojson(indent=2) }}

{% if prior_verdicts -%}
Prior attempt failed validation. Address each issue:
{% for v in prior_verdicts %}{% for i in v.issues %}
- [{{ i.severity }}] {{ i.location }} — {{ i.rule_violated }}: {{ i.explanation }}{% if i.suggested_fix %}
  Fix: {{ i.suggested_fix }}{% endif %}
{% endfor %}{% endfor %}
{%- endif %}

Return a structured Problem.
```

- [ ] **Step 2: `prompts/phase1/article_consistency.md.j2`**

```jinja
You are a consistency checker for KB articles.

Article to inspect:
{{ inputs.article | tojson(indent=2) }}

Return a Verdict. Set passed=false if the article contradicts itself or its associated problem.
```

- [ ] **Step 3: `prompts/phase2/turn_generator.md.j2`**

```jinja
You are simulating a {{ inputs.persona | default('support rep') }} turn.

Context:
{{ inputs | tojson(indent=2) }}

{% if prior_verdicts -%}
Prior attempt failed validation. Address each issue:
{% for v in prior_verdicts %}{% for i in v.issues %}
- [{{ i.severity }}] {{ i.location }} — {{ i.rule_violated }}: {{ i.explanation }}{% if i.suggested_fix %}
  Fix: {{ i.suggested_fix }}{% endif %}
{% endfor %}{% endfor %}
{%- endif %}

Return the next Turn.
```

- [ ] **Step 4: `prompts/phase2/turn_consistency.md.j2`**

```jinja
Inspect this turn for internal consistency with the prior conversation:
{{ inputs | tojson(indent=2) }}

Return a Verdict.
```

- [ ] **Step 5: `prompts/phase2/creative_noise.md.j2`**

```jinja
Apply one of the configured noise types to this turn:
{{ inputs.turn_content }}

Return a TurnModification with modified_content, noise_type, and rationale.
```

- [ ] **Step 6: Verify** — `uv run pytest -q` → no regressions (PromptRegistry will now discover these new templates).

- [ ] **Step 7: Commit**

```bash
git add prompts/phase1 prompts/phase2
git commit -m "feat: add placeholder Jinja templates for phase1/phase2 agents"
```

---

## Task 10: Integration test — full feedback loop with FakeChatModel

**Files:**
- Test: `tests/integration/test_feedback_loop.py`

- [ ] **Step 1: Write the test**

```python
import jinja2
import pytest
from pydantic import BaseModel

from csfd.agents.base import AgentContext, Issue, Verdict
from csfd.agents.checker import Checker
from csfd.agents.generator import Generator
from csfd.models.fake import FakeChatModel
from csfd.prompts.registry import PromptHandle


class _Problem(BaseModel):
    title: str
    description: str


def _h(src: str) -> PromptHandle:
    env = jinja2.Environment(autoescape=False)
    return PromptHandle(name="x", template=env.from_string(src), source=src, version="hhhhhhhhhhhh")


@pytest.mark.asyncio
async def test_full_feedback_loop_terminates_after_retry() -> None:
    """Run Generator → 3 parallel Checkers → aggregate → if any fail, retry Generator with issues."""
    # First-attempt artifact (will be rejected by consistency checker)
    bad = _Problem(title="x", description="Internal jira id INT-1234")
    good = _Problem(title="x", description="Customer-visible description")

    # Build a sequence of canned responses: the FakeChatModel returns whatever is keyed
    # to its current schema. We'll switch the structured map between attempts.
    llm_gen = FakeChatModel(structured={_Problem: bad})
    gen = Generator(
        name="problem_brainstorm",
        prompt=_h(
            "Generate {{ inputs.x }}."
            "{% for v in prior_verdicts %}{% for i in v.issues %}"
            "| FIX: {{ i.explanation }}"
            "{% endfor %}{% endfor %}"
        ),
        output_schema=_Problem,
        llm=llm_gen,
    )

    consistency_fail = Verdict(
        checker="consistency", passed=False,
        issues=[Issue(severity="error", location="description",
                       rule_violated="no_internal_ids",
                       explanation="Mentions internal jira id INT-1234")],
    )
    consistency_pass = Verdict(checker="consistency", passed=True)
    bg_pass = Verdict(checker="background", passed=True)
    sc_pass = Verdict(checker="scenario", passed=True)

    llm_cons = FakeChatModel(structured={Verdict: consistency_fail})
    llm_bg = FakeChatModel(structured={Verdict: bg_pass})
    llm_sc = FakeChatModel(structured={Verdict: sc_pass})

    cons = Checker(name="consistency", prompt=_h("check {{ inputs.artifact }}"), llm=llm_cons)
    bg = Checker(name="background", prompt=_h("check"), llm=llm_bg)
    sc = Checker(name="scenario", prompt=_h("check"), llm=llm_sc)

    # First attempt
    ctx = AgentContext(inputs={"x": "billing problem"})
    attempt1: BaseModel = await gen.invoke(ctx)
    v_cons = await cons.invoke(AgentContext(inputs={"artifact": attempt1.model_dump()}))
    v_bg = await bg.invoke(AgentContext(inputs={"artifact": attempt1.model_dump()}))
    v_sc = await sc.invoke(AgentContext(inputs={"artifact": attempt1.model_dump()}))
    verdicts = [v_cons, v_bg, v_sc]
    passed = all(v.passed for v in verdicts)
    assert passed is False
    assert any("INT-1234" in i.explanation for v in verdicts for i in v.issues)

    # Retry: flip the generator to good, flip consistency to pass
    llm_gen.structured = {_Problem: good}
    llm_cons.structured = {Verdict: consistency_pass}
    retry_ctx = AgentContext(
        inputs={"x": "billing problem"},
        prior_verdicts=verdicts,
        retry_attempt=1,
    )
    attempt2: BaseModel = await gen.invoke(retry_ctx)
    # The retry prompt MUST have included the fix instruction
    assert "INT-1234" in (gen.last_rendered_prompt or "")
    v_cons2 = await cons.invoke(AgentContext(inputs={"artifact": attempt2.model_dump()}))
    assert v_cons2.passed is True
```

- [ ] **Step 2: Verify** — `uv run pytest tests/integration/test_feedback_loop.py -v` → 1 pass; mypy + ruff clean; full suite green.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_feedback_loop.py
git commit -m "test: integration test for evaluator-optimizer feedback loop"
```

---

## Task 11: Full-suite verification

**Files:**
- Verify only; possibly add narrow type-ignores

- [ ] **Step 1: Run full suite**

```bash
uv run pytest -v
```
Expected: all tests (~73+) pass.

- [ ] **Step 2: Run mypy strict**

```bash
uv run mypy src tests
```
Expected: clean.

- [ ] **Step 3: Run ruff**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```
Expected: clean.

- [ ] **Step 4: Commit only if any cleanups landed**

If ruff format applied changes, commit:
```bash
git add -A
git commit -m "chore: ruff format after Plan 2 work"
```
Otherwise: no commit needed.

- [ ] **Step 5: Final summary**

`git log --oneline | head -15` showing the Plan 2 commits chained on Plan 1's final SHA `2e8ab0e`.

## Plan 2 completion criteria

- [ ] All 11 tasks complete; tests pass.
- [ ] Full suite green; mypy --strict clean; ruff clean.
- [ ] `csfd.agents` package exposes `Generator`, `Checker`, `CreativeNoise`, `Issue`, `Verdict`, `AgentContext`, `AgentRole`, `AgentFactory`, `TurnModification`, `record_agent_trace`.
- [ ] `FakeChatModel` supports `with_structured_output`.
- [ ] Placeholder Jinja templates exist for the agents Phase 3 and Phase 4 will instantiate.
- [ ] Integration test demonstrates the evaluator-optimizer feedback loop end-to-end with no LLM API key required.

Plan 2 done. Hand off to **Plan 3 — Phase 1 (KB Generation)**.
