# Turn-Based Dual-Agent Dialogue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase 2's single-shot resolution generator with a turn-by-turn dialogue between two information-asymmetric agents (customer + service), ended when either speaker flags `done`, then post-validated by a consistency agent that may edit the transcript in place.

**Architecture:** Approach A from the spec — each conversational turn is its own LangGraph node in `phase2_graph.py`. The customer agent sees only customer-observable problem fields (symptoms, impact, persona); the service agent sees only root-cause fields. Turns alternate via conditional edges until a `done` flag or a hard 20-turn cap; a consistency node then reviews the full transcript and may pass, pass-with-edits, or fail (full re-roll within the existing retry budget).

**Tech Stack:** Python 3.12+, LangGraph, Pydantic v2, Jinja2 prompts, SQLite (sync `Database` + `AsyncDatabase`), pytest. Spec: `docs/superpowers/specs/2026-05-27-turn-based-dialogue-design.md`.

---

## File Structure

**Source — modify:**
- `src/csfd/pipeline.py` — new schemas (`DialogueTurnOutput`, `IncomingRequestOutput`, `ConsistencyVerdict`), evolve `ResolutionOutput`, remove `ResolutionTurnOutput`, add `RESOLVED_DONE_REASONS` + `_derive_resolved` + `_assemble_resolution`.
- `src/csfd/graph/pipeline_graph.py` — add per-slot dialogue fields to `PipelineState`.
- `src/csfd/settings.py` — add `DialogueConfig`, replace `TicketsConfig.turns_per_type` with `TicketsConfig.dialogue`.
- `src/csfd/graph/phase2_graph.py` — full rewrite of the per-slot region (nodes + routers + builder).
- `src/csfd/agents/factory.py` — update `_ROLE_FALLBACK` for the new node names.
- `src/csfd/models/fake.py` — add sequenced structured-output support for multi-call tests.
- `src/csfd/graph/rendering.py` — update the stub prompt-name list used by graph rendering.
- `config/default.yaml` — remove `tickets.turns_per_type`, add `tickets.dialogue.turn_cap`.

**Prompts — create:**
- `prompts/phase2/incoming_request.md.j2`
- `prompts/phase2/customer_turn.md.j2`
- `prompts/phase2/agent_turn.md.j2`
- `prompts/phase2/conversation_consistency_check.md.j2`

**Prompts — delete:**
- `prompts/phase2/resolution_generator.md.j2`
- `prompts/phase2/resolution_combined_check.md.j2`

**Tests — create:**
- `tests/unit/test_dialogue_state.py`
- `tests/unit/test_consistency_verdict.py`
- `tests/integration/test_phase2_dialogue_happy_path.py`
- `tests/integration/test_phase2_dialogue_cap_hit.py`
- `tests/integration/test_phase2_dialogue_consistency_retry.py`
- `tests/integration/test_phase2_dialogue_consistency_edit.py`

**Tests — modify (remove old prompt names / `turns_per_type`):**
- `tests/unit/test_phase2_graph.py` (rewrite routers section)
- `tests/unit/test_phase1_graph.py`, `tests/unit/test_pipeline_graph.py`, `tests/unit/test_settings.py`, `tests/unit/test_prompt_render_smoke.py`
- `tests/integration/test_pipeline_retry.py`, `tests/integration/test_phase1_dedup.py`, `tests/integration/test_pipeline_deterministic.py`

**Docs — modify:**
- `README.md` (tickets section: turn count is now emergent; cost note).

---

## Task 1: New dialogue schemas + resolved-flag helper

**Files:**
- Modify: `src/csfd/pipeline.py` (schemas near lines 47-75; helpers near line 80+)
- Test: `tests/unit/test_dialogue_state.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_dialogue_state.py`:

```python
"""Unit tests for dialogue schemas, resolved-flag derivation, and transcript assembly."""

from __future__ import annotations

import pytest

from csfd.pipeline import (
    DialogueTurnOutput,
    IncomingRequestOutput,
    ResolutionOutput,
    _assemble_resolution,
    _derive_resolved,
)


def test_dialogue_turn_output_fields() -> None:
    t = DialogueTurnOutput(speaker="customer", content="hi", done=False)
    assert t.speaker == "customer"
    assert t.done is False
    assert t.done_reason is None


def test_incoming_request_output_fields() -> None:
    ir = IncomingRequestOutput(subject="S", body="B")
    assert ir.subject == "S"
    assert ir.body == "B"


@pytest.mark.parametrize(
    ("end_reason", "last_reason", "expected"),
    [
        ("agent_done", "resolved", True),
        ("customer_done", "customer_satisfied", True),
        ("agent_done", "issue_fixed", True),
        ("agent_done", "closed", True),
        ("agent_done", "escalation", False),
        ("customer_done", "customer_frustrated", False),
        ("agent_done", None, False),
        ("cap_hit", "resolved", False),
        ("cap_hit", None, False),
    ],
)
def test_derive_resolved(end_reason: str, last_reason: str | None, expected: bool) -> None:
    assert _derive_resolved(end_reason, last_reason) is expected


def test_assemble_resolution_builds_output() -> None:
    turns = [
        DialogueTurnOutput(speaker="customer", content="opening", done=False),
        DialogueTurnOutput(speaker="agent", content="fixed it", done=True, done_reason="resolved"),
    ]
    res = _assemble_resolution(
        subject="Sub", body="opening", turns=turns, end_reason="agent_done"
    )
    assert isinstance(res, ResolutionOutput)
    assert res.subject == "Sub"
    assert res.body == "opening"
    assert res.turns == turns
    assert res.end_reason == "agent_done"
    assert res.resolved is True


def test_assemble_resolution_cap_hit_is_unresolved() -> None:
    turns = [DialogueTurnOutput(speaker="customer", content="x", done=False)]
    res = _assemble_resolution(subject="S", body="x", turns=turns, end_reason="cap_hit")
    assert res.resolved is False
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/unit/test_dialogue_state.py -v`
Expected: FAIL with `ImportError` (`DialogueTurnOutput` etc. not defined).

- [ ] **Step 3: Implement schemas + helpers in `src/csfd/pipeline.py`**

Replace the existing `ResolutionTurnOutput` and `ResolutionOutput` classes (currently lines 59-75) with:

```python
class DialogueTurnOutput(BaseModel):
    speaker: Literal["customer", "agent"]
    content: str
    done: bool = False
    done_reason: str | None = None


class IncomingRequestOutput(BaseModel):
    subject: str
    body: str


class ResolutionOutput(BaseModel):
    subject: str
    body: str
    turns: list[DialogueTurnOutput] = Field(default_factory=list)
    resolved: bool = False
    end_reason: Literal["customer_done", "agent_done", "cap_hit"] = "agent_done"


class ConsistencyVerdict(BaseModel):
    status: Literal["pass", "pass_with_edits", "fail"]
    issues: list[str] = Field(default_factory=list)
    edited_turns: list[DialogueTurnOutput] | None = None
    edited_subject: str | None = None
    edited_body: str | None = None
```

Then add the helpers right after the schema block (before the "Pure helpers" section near current line 78):

```python
RESOLVED_DONE_REASONS = frozenset(
    {"resolved", "customer_satisfied", "issue_fixed", "closed"}
)


def _derive_resolved(end_reason: str, last_done_reason: str | None) -> bool:
    """True iff the conversation ended naturally on a resolved-style reason."""
    if end_reason == "cap_hit":
        return False
    return last_done_reason in RESOLVED_DONE_REASONS


def _assemble_resolution(
    *,
    subject: str,
    body: str,
    turns: list[DialogueTurnOutput],
    end_reason: Literal["customer_done", "agent_done", "cap_hit"],
) -> ResolutionOutput:
    """Build a ResolutionOutput from accumulated dialogue state, deriving `resolved`."""
    last_reason = turns[-1].done_reason if turns else None
    return ResolutionOutput(
        subject=subject,
        body=body,
        turns=turns,
        resolved=_derive_resolved(end_reason, last_reason),
        end_reason=end_reason,
    )
```

Also update the module docstring bullet at line 9 from `ResolutionTurnOutput` to `DialogueTurnOutput`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/unit/test_dialogue_state.py -v`
Expected: PASS (all 5 test functions, parametrized cases green).

- [ ] **Step 5: Commit**

```bash
git add src/csfd/pipeline.py tests/unit/test_dialogue_state.py
git commit -m "feat(phase2): dialogue schemas + resolved-flag derivation"
```

---

## Task 2: ConsistencyVerdict edit-application helper

**Files:**
- Modify: `src/csfd/pipeline.py` (add `_apply_consistency_edits` after `_assemble_resolution`)
- Test: `tests/unit/test_consistency_verdict.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_consistency_verdict.py`:

```python
"""Unit tests for ConsistencyVerdict and edit application."""

from __future__ import annotations

from csfd.pipeline import (
    ConsistencyVerdict,
    DialogueTurnOutput,
    ResolutionOutput,
    _apply_consistency_edits,
)


def _draft() -> ResolutionOutput:
    return ResolutionOutput(
        subject="orig subj",
        body="orig body",
        turns=[
            DialogueTurnOutput(speaker="customer", content="orig body", done=False),
            DialogueTurnOutput(speaker="agent", content="ok", done=True, done_reason="resolved"),
        ],
        resolved=True,
        end_reason="agent_done",
    )


def test_pass_returns_draft_unchanged() -> None:
    draft = _draft()
    verdict = ConsistencyVerdict(status="pass")
    out = _apply_consistency_edits(draft, verdict)
    assert out is draft


def test_pass_with_edits_replaces_provided_fields() -> None:
    draft = _draft()
    new_turns = [
        DialogueTurnOutput(speaker="customer", content="edited body", done=False),
        DialogueTurnOutput(speaker="agent", content="edited", done=True, done_reason="resolved"),
    ]
    verdict = ConsistencyVerdict(
        status="pass_with_edits",
        edited_subject="new subj",
        edited_body="edited body",
        edited_turns=new_turns,
    )
    out = _apply_consistency_edits(draft, verdict)
    assert out.subject == "new subj"
    assert out.body == "edited body"
    assert out.turns == new_turns
    # resolved is re-derived from the edited turns + original end_reason
    assert out.resolved is True
    assert out.end_reason == "agent_done"


def test_pass_with_edits_partial_keeps_originals() -> None:
    draft = _draft()
    verdict = ConsistencyVerdict(status="pass_with_edits", edited_subject="only subj")
    out = _apply_consistency_edits(draft, verdict)
    assert out.subject == "only subj"
    assert out.body == "orig body"
    assert out.turns == draft.turns
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/unit/test_consistency_verdict.py -v`
Expected: FAIL with `ImportError` (`_apply_consistency_edits` not defined).

- [ ] **Step 3: Implement `_apply_consistency_edits` in `src/csfd/pipeline.py`**

Add after `_assemble_resolution`:

```python
def _apply_consistency_edits(
    draft: ResolutionOutput, verdict: "ConsistencyVerdict"
) -> ResolutionOutput:
    """Return the draft unchanged on pass, or a new draft with the verdict's edits applied.

    Only fields the verdict explicitly sets are replaced; `resolved` is re-derived
    from the (possibly edited) turns against the original `end_reason`.
    """
    if verdict.status != "pass_with_edits":
        return draft
    subject = verdict.edited_subject if verdict.edited_subject is not None else draft.subject
    body = verdict.edited_body if verdict.edited_body is not None else draft.body
    turns = verdict.edited_turns if verdict.edited_turns is not None else draft.turns
    return _assemble_resolution(
        subject=subject, body=body, turns=turns, end_reason=draft.end_reason
    )
```

Note: `ConsistencyVerdict` is already defined in Task 1; the forward-reference string in the signature is only for readability and may be a plain annotation since the class is defined above.

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/unit/test_consistency_verdict.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/csfd/pipeline.py tests/unit/test_consistency_verdict.py
git commit -m "feat(phase2): consistency-verdict edit application"
```

---

## Task 3: Extend PipelineState with per-slot dialogue fields

**Files:**
- Modify: `src/csfd/graph/pipeline_graph.py` (`PipelineState`, after the Phase 2 progress block near line 96)

- [ ] **Step 1: Add fields**

In `src/csfd/graph/pipeline_graph.py`, locate the `# --- Phase 2 progress ---` block in `PipelineState` and add a dialogue sub-block immediately after `slot_index: int = 0`:

```python
    # --- per-slot dialogue progress (reset in commit_dialogue_node) ---
    current_dialogue_turns: list[DialogueTurnOutput] = Field(default_factory=list)
    dialogue_last_speaker: Literal["customer", "agent"] | None = None
    dialogue_done: bool = False
    dialogue_end_reason: Literal["customer_done", "agent_done", "cap_hit"] | None = None
```

Update the imports at the top of the file:
- Add `from typing import Literal` (or extend the existing `from typing import Any` import to `from typing import Any, Literal`).
- Change `from csfd.pipeline import (ProblemBrainstormOutput, ResolutionOutput, _acompute_run_stats)` to also import `DialogueTurnOutput`.

- [ ] **Step 2: Verify import + model build**

Run: `uv run python -c "from csfd.graph.pipeline_graph import PipelineState; print('ok')"`
Expected: prints `ok` (no import or Pydantic errors).

- [ ] **Step 3: Run the existing pipeline-graph unit test (smoke)**

Run: `uv run pytest tests/unit/test_pipeline_graph.py -v`
Expected: may FAIL only on `turns_per_type`/old-prompt references (addressed in Task 10). If it fails for any *other* reason, stop and investigate. The `PipelineState` construction itself must not error.

- [ ] **Step 4: Commit**

```bash
git add src/csfd/graph/pipeline_graph.py
git commit -m "feat(phase2): add per-slot dialogue fields to PipelineState"
```

---

## Task 4: Settings — DialogueConfig replaces turns_per_type

**Files:**
- Modify: `src/csfd/settings.py` (`TicketsConfig` near line 51; add `DialogueConfig`)
- Test: `tests/unit/test_settings.py` (lines 47, 203, 215 reference `turns_per_type`)

- [ ] **Step 1: Update the failing settings test first**

In `tests/unit/test_settings.py`:
- Line 47: replace the `turns_per_type: {l1: 2}` YAML line with `dialogue: {turn_cap: 12}`.
- Line ~203: replace `assert s.tickets.turns_per_type` with `assert s.tickets.dialogue.turn_cap == 12` (use whatever value the surrounding fixture sets; if the fixture is the default config, assert `== 20`).
- Lines ~215: the assertion `types <= set(s.tickets.turns_per_type)` checks ticket-type coverage via `turns_per_type` keys — change it to use `s.tickets.type_proportions` keys instead: `assert types <= set(s.tickets.type_proportions)`.

Read the surrounding test bodies before editing so the new assertions match the fixture in scope.

- [ ] **Step 2: Run, verify failure**

Run: `uv run pytest tests/unit/test_settings.py -v`
Expected: FAIL — `TicketsConfig` has no `dialogue` field yet (validation error), and `DialogueConfig` import / attribute access fails.

- [ ] **Step 3: Implement `DialogueConfig` and edit `TicketsConfig`**

In `src/csfd/settings.py`, add before `class TicketsConfig`:

```python
class DialogueConfig(BaseModel):
    turn_cap: int = 20
```

In `class TicketsConfig`, remove the line `turns_per_type: dict[str, int]` and add:

```python
    dialogue: DialogueConfig = Field(default_factory=DialogueConfig)
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/unit/test_settings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/settings.py tests/unit/test_settings.py
git commit -m "feat(config): DialogueConfig.turn_cap replaces tickets.turns_per_type"
```

---

## Task 5: Update config files

**Files:**
- Modify: `config/default.yaml` (tickets section)

- [ ] **Step 1: Edit `config/default.yaml`**

Remove the entire `turns_per_type:` block (the key and its four child lines `docs_request: 2`, `l1: 3`, `l2: 5`, `l3: 7`). Add under `tickets:` (after `assignment_strategy:`):

```yaml
  dialogue:
    turn_cap: 20
```

- [ ] **Step 2: Verify config loads**

Run: `uv run python -c "from csfd.settings import load_settings; s = load_settings(); print(s.tickets.dialogue.turn_cap)"`
Expected: prints `20`.

- [ ] **Step 3: Confirm no profile overlay sets turns_per_type**

Run: `grep -rn "turns_per_type" config/`
Expected: no output (profiles never set it; confirmed at planning time).

- [ ] **Step 4: Commit**

```bash
git add config/default.yaml
git commit -m "feat(config): default.yaml dialogue.turn_cap=20, drop turns_per_type"
```

---

## Task 6: AgentFactory role fallbacks for new node names

**Files:**
- Modify: `src/csfd/agents/factory.py` (`_ROLE_FALLBACK` near line 19)
- Test: `tests/unit/test_agent_factory.py` (add a case)

- [ ] **Step 1: Write a failing test**

In `tests/unit/test_agent_factory.py`, add (adapt to the file's existing fixture style):

```python
def test_role_fallback_routes_new_dialogue_nodes() -> None:
    from csfd.agents.factory import _ROLE_FALLBACK

    assert _ROLE_FALLBACK["incoming_request_generator"] == "generator"
    assert _ROLE_FALLBACK["customer_turn_generator"] == "generator"
    assert _ROLE_FALLBACK["agent_turn_generator"] == "generator"
    assert _ROLE_FALLBACK["conversation_consistency_check"] == "combined_checker"
    assert "resolution_generator" not in _ROLE_FALLBACK
    assert "combined_resolution_check" not in _ROLE_FALLBACK
```

- [ ] **Step 2: Run, verify failure**

Run: `uv run pytest tests/unit/test_agent_factory.py::test_role_fallback_routes_new_dialogue_nodes -v`
Expected: FAIL (KeyError / assertion error).

- [ ] **Step 3: Edit `_ROLE_FALLBACK`**

Replace the dict in `src/csfd/agents/factory.py` with:

```python
_ROLE_FALLBACK: dict[str, str] = {
    "problem_brainstorm": "generator",
    "incoming_request_generator": "generator",
    "customer_turn_generator": "generator",
    "agent_turn_generator": "generator",
    "combined_problem_check": "combined_checker",
    "conversation_consistency_check": "combined_checker",
}
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/unit/test_agent_factory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/agents/factory.py tests/unit/test_agent_factory.py
git commit -m "feat(phase2): route new dialogue node names to agent buckets"
```

---

## Task 7: FakeChatModel sequenced structured output

**Files:**
- Modify: `src/csfd/models/fake.py`
- Test: `tests/unit/test_fake_structured.py` (add a case)

Rationale: turn-based tests issue many calls that all bind the **same** schema (`DialogueTurnOutput`). The current `structured` dict returns one fixed value per schema. We add an optional `structured_seq` queue that, when present for a schema, is consumed front-to-back; the existing `structured` dict remains the fallback for schemas without a queue.

- [ ] **Step 1: Write a failing test**

In `tests/unit/test_fake_structured.py`, add:

```python
def test_structured_seq_consumed_in_order() -> None:
    from pydantic import BaseModel

    from csfd.models.fake import FakeChatModel

    class S(BaseModel):
        v: int

    fake = FakeChatModel(structured_seq={S: [S(v=1), S(v=2)]})
    runnable = fake.with_structured_output(S)
    assert runnable.invoke("x").v == 1
    assert runnable.invoke("x").v == 2


def test_structured_seq_falls_back_to_structured_when_empty() -> None:
    from pydantic import BaseModel

    from csfd.models.fake import FakeChatModel

    class S(BaseModel):
        v: int

    fake = FakeChatModel(structured={S: S(v=9)}, structured_seq={S: []})
    assert fake.with_structured_output(S).invoke("x").v == 9
```

- [ ] **Step 2: Run, verify failure**

Run: `uv run pytest tests/unit/test_fake_structured.py -v`
Expected: FAIL (`FakeChatModel` has no `structured_seq` field).

- [ ] **Step 3: Implement in `src/csfd/models/fake.py`**

Add the field next to `structured` (after line 17):

```python
    structured_seq: dict[type[BaseModel], list[BaseModel]] = Field(default_factory=dict)
```

Replace the body of `with_structured_output` (lines 62-71) with:

```python
        canned = self.structured
        seq = self.structured_seq

        def _return_canned(_inputs: Any) -> BaseModel:
            queue = seq.get(schema)
            if queue:
                return queue.pop(0)
            if schema not in canned:
                raise KeyError(
                    f"FakeChatModel has no canned structured output for {schema.__name__}"
                )
            return canned[schema]

        return RunnableLambda(_return_canned)
```

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/unit/test_fake_structured.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/csfd/models/fake.py tests/unit/test_fake_structured.py
git commit -m "test(fake): sequenced structured-output queue for multi-call flows"
```

---

## Task 8: Author the four new Phase 2 prompts

**Files:**
- Create: `prompts/phase2/incoming_request.md.j2`
- Create: `prompts/phase2/customer_turn.md.j2`
- Create: `prompts/phase2/agent_turn.md.j2`
- Create: `prompts/phase2/conversation_consistency_check.md.j2`

All four are Jinja2 templates rendered with an `inputs` dict (and `prior_verdicts` for generators, matching the existing convention in `resolution_generator.md.j2`). The renderer passes `inputs` as a dict; reference fields as `inputs.<field>`.

- [ ] **Step 1: Create `prompts/phase2/incoming_request.md.j2`**

```jinja
You are roleplaying a customer of {{ inputs.company_name | default('the company') }} writing the FIRST message of a support ticket. You only know what you have personally observed — you do NOT know the underlying cause.

What you are experiencing (your symptoms — this is ALL you know):
{% for s in inputs.symptoms %}- {{ s }}
{% endfor %}
Impact on you: {{ inputs.customer_impact }}
Ticket category (for routing only): {{ inputs.category }}

Your identity:
- Name: {{ inputs.customer_name }}
- Tier: {{ inputs.customer_tier }}
- Tone: {{ inputs.customer_tone }}

Rules:
- Write ONLY from your symptoms. Never name a root cause, internal component, or fix — you don't know them.
- Be partial or vague where a real customer would be: you may omit details you wouldn't think to mention.
- Match your tone ({{ inputs.customer_tone }}) consistently.
- Keep it to a natural opening message: a subject line (≤80 chars) and a body of 2–5 sentences.

Output contract — return:
- `subject`: short email-style subject line.
- `body`: your opening message as a complete email (not a summary).
```

- [ ] **Step 2: Create `prompts/phase2/customer_turn.md.j2`**

```jinja
You are roleplaying the SAME customer of {{ inputs.company_name | default('the company') }} continuing a support conversation. You only know your own symptoms — never the root cause or the fix unless the agent has just told you.

Your symptoms (all you know):
{% for s in inputs.symptoms %}- {{ s }}
{% endfor %}
Impact on you: {{ inputs.customer_impact }}
Your tone (keep consistent): {{ inputs.customer_tone }}

Conversation so far (turn {{ inputs.turn_index }} of at most {{ inputs.turn_cap }}):
{% for t in inputs.conversation_so_far %}{{ t.speaker }}: {{ t.content }}
{% endfor %}

How to respond:
- Answer the agent's latest message using ONLY your symptoms and what they've told you. Don't invent technical facts you couldn't know.
- If the agent's question is ambiguous, it is realistic to misunderstand it or ask what they mean — this is allowed and encouraged occasionally.
- Short conversations are correct for simple problems. If the agent's solution clearly addresses what you described, acknowledge it and set `done=true`. Don't invent extra questions.
- Set `done=true` when satisfied (`done_reason="customer_satisfied"`), when told it's fixed (`done_reason="resolved"`), or when leaving unhappy (`done_reason="customer_frustrated"`). Otherwise `done=false`.

Output contract — return one turn:
- `speaker`: always "customer".
- `content`: your reply (1–6 sentences).
- `done`: boolean (see above).
- `done_reason`: short tag when done, else null.
```

- [ ] **Step 3: Create `prompts/phase2/agent_turn.md.j2`**

```jinja
You are a {{ inputs.ticket_type }} support agent ({{ inputs.agent_name }}) at {{ inputs.company_name | default('the company') }}. You have internal diagnostic knowledge the customer does NOT have. You do NOT directly see a "symptoms" list — read the customer's symptoms from their messages.

Internal knowledge (do not paste verbatim; use it to guide diagnosis):
- Problem title: {{ inputs.problem.title }}
- Summary: {{ inputs.problem.summary }}
- Background: {{ inputs.problem.background }}
- Category: {{ inputs.problem.category }}
- Fault domain: {{ inputs.problem.fault_domain }}
- Root cause:
{% for rc in inputs.problem.root_cause %}  - {{ rc }}
{% endfor %}
- Resolution guidance for this ticket type: {{ inputs.problem.resolution_hint }}

Conversation so far (turn {{ inputs.turn_index }} of at most {{ inputs.turn_cap }}):
{% for t in inputs.conversation_so_far %}{{ t.speaker }}: {{ t.content }}
{% endfor %}

How to respond:
- Read the customer's described symptoms; ask a clarifying question when they are ambiguous rather than assuming.
- Reveal your diagnostic reasoning progressively and guide the customer toward the resolution guidance.
- Short conversations are correct for simple problems. If the symptoms are clear enough to answer in one turn, do so. Do not pad with unnecessary clarifying questions when the description is already sufficient.
- Set `done=true` when the issue is resolved (`done_reason="resolved"`), the customer is satisfied (`done_reason="customer_satisfied"`), or you must close/escalate (`done_reason="escalation"` or `"closed"`). Otherwise `done=false`.
- Stay within your tier: a {{ inputs.ticket_type }} agent must not promise actions outside its scope.

{% if prior_verdicts -%}
A prior version of this conversation failed consistency review. Address each issue:
{% for v in prior_verdicts %}{% for i in v.issues %}- {{ i }}
{% endfor %}{% endfor %}
{%- endif %}

Output contract — return one turn:
- `speaker`: always "agent".
- `content`: your reply (1–6 sentences).
- `done`: boolean (see above).
- `done_reason`: short tag when done, else null.
```

Note: `prior_verdicts` here is rendered as a list of objects each carrying an `issues` list of strings; the consistency verdict's `issues` is `list[str]`. The agent-turn node only passes prior consistency issues on a retry (see Task 9 node code) — wrap each issue string so `i` is a string. If the node passes raw strings, simplify the loop to `{% for i in prior_verdicts %}- {{ i }}`. Use the raw-strings form to match the node implementation in Task 9:

Replace the `{% if prior_verdicts %}` block above with:

```jinja
{% if prior_issues -%}
A prior version of this conversation failed consistency review. Address each issue:
{% for i in prior_issues %}- {{ i }}
{% endfor %}
{%- endif %}
```

- [ ] **Step 4: Create `prompts/phase2/conversation_consistency_check.md.j2`**

```jinja
You are a consistency reviewer for synthetic customer-service conversations at {{ inputs.company_name | default('the company') }}. You see BOTH the customer-facing view and the internal root cause. Judge the full transcript and either pass it, pass it with edits, or fail it.

Problem (full context):
- Title: {{ inputs.problem.title }}
- Complexity: {{ inputs.problem.complexity }}
- Customer-observable symptoms:
{% for s in inputs.problem.symptoms %}  - {{ s }}
{% endfor %}
- Root cause:
{% for rc in inputs.problem.root_cause %}  - {{ rc }}
{% endfor %}
- Resolution guidance: {{ inputs.problem.resolution_hint }}
- Customer tone (expected, consistent): {{ inputs.customer_tone }}

Conversation to review:
subject: {{ inputs.candidate.subject }}
body: {{ inputs.candidate.body }}
end_reason: {{ inputs.candidate.end_reason }}
turns:
{% for t in inputs.candidate.turns %}{{ loop.index }}. {{ t.speaker }}{% if t.done %} [done: {{ t.done_reason }}]{% endif %}: {{ t.content }}
{% endfor %}

Checks (flag and, where fixable by light editing, correct in place):
- (a) The customer never uses root-cause language they could not know.
- (b) The agent's diagnostic path is plausible given only the symptoms in the messages.
- (c) Information ping-pong is appropriate to complexity: `simple` problems may resolve in 2–3 turns; `medium`/`complex` should show progressive discovery. Do NOT penalise a short conversation for a simple problem.
- (d) Tone is consistent with {{ inputs.customer_tone }} across all customer turns.
- (e) No contradictions between turns.
- (f) `end_reason` and the implied resolution match what the conversation actually achieved.

Output contract — return:
- `status`: "pass", "pass_with_edits", or "fail".
- `issues`: list of short strings describing each problem found (empty on a clean pass).
- `edited_turns`: only when status is "pass_with_edits" — the full corrected turn list (same shape as input turns). Null otherwise.
- `edited_subject` / `edited_body`: only when you changed them; null otherwise.
Use "fail" only for problems too structural to fix by editing (the conversation will be fully regenerated).
```

- [ ] **Step 5: Verify all four templates parse**

Run:
```bash
uv run python -c "
from pathlib import Path
import jinja2
env = jinja2.Environment(autoescape=False)
for p in ['incoming_request','customer_turn','agent_turn','conversation_consistency_check']:
    env.from_string(Path(f'prompts/phase2/{p}.md.j2').read_text())
print('all parse ok')
"
```
Expected: prints `all parse ok` (no `TemplateSyntaxError`).

- [ ] **Step 6: Commit**

```bash
git add prompts/phase2/incoming_request.md.j2 prompts/phase2/customer_turn.md.j2 prompts/phase2/agent_turn.md.j2 prompts/phase2/conversation_consistency_check.md.j2
git commit -m "feat(phase2): turn-based dialogue + consistency prompts"
```

---

## Task 9: Rewrite the Phase 2 subgraph (nodes, routers, builder)

**Files:**
- Modify: `src/csfd/graph/phase2_graph.py` (replace `generate_resolution_node`, `validate_resolution_node`, their routers, and the builder wiring; keep `build_allocation_plan_node`, `_bump_retry_node`, `_mark_exhausted_node`, `_mark_validation_skipped_node`, `_route_after_build_allocation_plan`, `_route_after_commit_resolution`)
- Test: `tests/unit/test_phase2_graph.py` (rewrite router section)

This is the largest task. Implement routers first (TDD), then the nodes, then the builder.

### 9a. Routers + new constants

- [ ] **Step 1: Rewrite the router section of `tests/unit/test_phase2_graph.py`**

Replace the imports from `csfd.graph.phase2_graph` (lines 20-25) with:

```python
from csfd.graph.phase2_graph import (
    _route_after_agent_turn,
    _route_after_build_allocation_plan,
    _route_after_commit_resolution,
    _route_after_customer_turn,
    _route_after_validate_conversation,
    build_phase2_subgraph,
)
```

In `_build_stub_settings()` (around line 88) remove the `turns_per_type={...}` argument and add `dialogue=DialogueConfig(turn_cap=20)` (import `DialogueConfig` from `csfd.settings`).

In `_build_stub_factory()` replace the prompt-handle keys (lines 115-116) `"phase2.resolution_generator"` / `"phase2.resolution_combined_check"` with:

```python
        "phase2.incoming_request": handle,
        "phase2.customer_turn": handle,
        "phase2.agent_turn": handle,
        "phase2.conversation_consistency_check": handle,
```

Replace the router test functions (`test_route_after_generate_*`, `test_route_after_validate_*`) with:

```python
def _dialogue_turns(n: int) -> list[Any]:
    from csfd.pipeline import DialogueTurnOutput

    return [
        DialogueTurnOutput(speaker="customer" if i % 2 == 0 else "agent", content=f"t{i}")
        for i in range(n)
    ]


def test_route_after_agent_turn_done() -> None:
    state = _make_state(dialogue_done=True, current_dialogue_turns=_dialogue_turns(4))
    assert _route_after_agent_turn(state) == "consistency"


def test_route_after_agent_turn_continue() -> None:
    state = _make_state(dialogue_done=False, current_dialogue_turns=_dialogue_turns(4))
    assert _route_after_agent_turn(state) == "customer"


def test_route_after_agent_turn_cap() -> None:
    state = _make_state(dialogue_done=False, current_dialogue_turns=_dialogue_turns(20))
    assert _route_after_agent_turn(state) == "cap"


def test_route_after_customer_turn_done() -> None:
    state = _make_state(dialogue_done=True, current_dialogue_turns=_dialogue_turns(3))
    assert _route_after_customer_turn(state) == "consistency"


def test_route_after_customer_turn_continue() -> None:
    state = _make_state(dialogue_done=False, current_dialogue_turns=_dialogue_turns(3))
    assert _route_after_customer_turn(state) == "agent"


def test_route_after_customer_turn_cap() -> None:
    state = _make_state(dialogue_done=False, current_dialogue_turns=_dialogue_turns(20))
    assert _route_after_customer_turn(state) == "cap"


def test_route_after_validate_pass() -> None:
    state = _make_state(
        last_verdict=Verdict(checker="x", passed=True, issues=[]), retry_attempt=0, max_retries=2
    )
    assert _route_after_validate_conversation(state) == "pass"


def test_route_after_validate_retry() -> None:
    state = _make_state(
        last_verdict=Verdict(checker="x", passed=False, issues=[]), retry_attempt=1, max_retries=3
    )
    assert _route_after_validate_conversation(state) == "retry"


def test_route_after_validate_exhausted() -> None:
    state = _make_state(
        last_verdict=Verdict(checker="x", passed=False, issues=[]), retry_attempt=2, max_retries=2
    )
    assert _route_after_validate_conversation(state) == "exhausted"
```

The cap value `20` matches `DialogueConfig.turn_cap` in the stub settings; the routers read the cap from a value closed over at build time, but for unit testing the routers directly we pass the cap via a module constant fallback. To keep the routers unit-testable without a settings object, the routers take the cap from `state` — add `dialogue_turn_cap: int = 20` to `PipelineState` in this task (see Step 3) so routers can read `state.dialogue_turn_cap`.

- [ ] **Step 2: Run, verify failure**

Run: `uv run pytest tests/unit/test_phase2_graph.py -v`
Expected: FAIL (router functions not defined / renamed).

- [ ] **Step 3: Add `dialogue_turn_cap` to `PipelineState`**

In `src/csfd/graph/pipeline_graph.py`, in the dialogue sub-block added in Task 3, add:

```python
    dialogue_turn_cap: int = 20
```

This is set by `build_allocation_plan_node` from settings so the routers (which only receive `state`) can read it.

- [ ] **Step 4: Implement the routers in `src/csfd/graph/phase2_graph.py`**

Remove `_route_after_generate_resolution` and `_route_after_validate_resolution`. Add:

```python
def _route_after_agent_turn(state: PipelineState) -> Literal["customer", "consistency", "cap"]:
    if state.dialogue_done:
        return "consistency"
    if len(state.current_dialogue_turns) >= state.dialogue_turn_cap:
        return "cap"
    return "customer"


def _route_after_customer_turn(state: PipelineState) -> Literal["agent", "consistency", "cap"]:
    if state.dialogue_done:
        return "consistency"
    if len(state.current_dialogue_turns) >= state.dialogue_turn_cap:
        return "cap"
    return "agent"


def _route_after_validate_conversation(
    state: PipelineState,
) -> Literal["pass", "retry", "exhausted"]:
    assert state.last_verdict is not None, "validate_conversation_node must set last_verdict"
    if state.last_verdict.passed:
        return "pass"
    if state.retry_attempt >= state.max_retries:
        return "exhausted"
    return "retry"
```

- [ ] **Step 5: Run router tests, verify pass**

Run: `uv run pytest tests/unit/test_phase2_graph.py -k route -v`
Expected: PASS for all router tests. (Builder-shape tests may still fail until 9b/9c — that's fine; re-run the whole file at the end of Task 9.)

### 9b. Nodes

- [ ] **Step 6: Rewrite the node section of `src/csfd/graph/phase2_graph.py`**

Update imports at the top:

```python
from csfd.pipeline import (
    ConsistencyVerdict,
    DialogueTurnOutput,
    IncomingRequestOutput,
    ResolutionOutput,
    _apply_consistency_edits,
    _assemble_resolution,
)
```

Keep `from csfd.agents.base import AgentContext, Verdict` (Verdict still used by the checker adapter path). Add a small render helper near the top of the node section:

```python
def _render_history(turns: list[DialogueTurnOutput]) -> list[dict[str, str]]:
    """Render accumulated turns into the simple {speaker, content} dicts the prompts expect."""
    return [{"speaker": t.speaker, "content": t.content} for t in turns]
```

Replace `build_allocation_plan_node`'s return (line 122) to also seed the cap:

```python
    return {"plan_slots": pydantic_slots, "slot_index": 0, "dialogue_turn_cap": settings.tickets.dialogue.turn_cap}
```

Delete `generate_resolution_node` and `validate_resolution_node`. Add the new nodes:

```python
def _customer_inputs(state: PipelineState, slot: _PlanSlot) -> dict[str, Any]:
    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    return {
        "company_name": state.company.name,
        "ticket_type": slot.ticket_type.value,
        "customer_name": f"Customer-{slot.tier}-{slot.index:04d}",
        "customer_tier": slot.tier,
        "customer_tone": slot.tone,
        "symptoms": problem.symptoms,
        "customer_impact": problem.customer_impact,
        "category": problem.category,
        "turn_cap": state.dialogue_turn_cap,
    }


def _agent_inputs(state: PipelineState, slot: _PlanSlot) -> dict[str, Any]:
    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    return {
        "company_name": state.company.name,
        "ticket_type": slot.ticket_type.value,
        "agent_name": f"Agent-{slot.ticket_type.value}-{slot.index:04d}",
        "problem": {
            "title": problem.title,
            "summary": problem.summary,
            "background": problem.background,
            "category": problem.category,
            "fault_domain": problem.fault_domain,
            "root_cause": problem.root_cause,
            "resolution_hint": problem.resolution_hints.get(slot.ticket_type.value, ""),
        },
        "turn_cap": state.dialogue_turn_cap,
    }


async def generate_incoming_request_node(
    state: PipelineState, *, factory: AgentFactory, db: Database, adb: AsyncDatabase, settings: AppSettings
) -> dict[str, Any]:
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    inputs = _customer_inputs(state, slot)

    link = ParentLink()
    generator = factory.build_generator(
        name="incoming_request_generator",
        prompt_name="phase2.incoming_request",
        output_schema_factory=lambda: IncomingRequestOutput,
    )
    traced = TracingAdapter(
        inner=generator, db=db, adb=adb, run_id=state.run_id,
        node_name="incoming_request_generator", artifact_type="resolution",
        artifact_id=ticket_uid, role="generator", parent_link=link,
    )
    result = await traced.invoke(AgentContext(inputs=inputs, retry_attempt=state.retry_attempt))
    assert isinstance(result, IncomingRequestOutput)
    turn0 = DialogueTurnOutput(speaker="customer", content=result.body, done=False)
    return {
        "current_resolution_draft": ResolutionOutput(subject=result.subject, body=result.body),
        "current_dialogue_turns": [turn0],
        "dialogue_last_speaker": "customer",
        "dialogue_done": False,
        "dialogue_end_reason": None,
        "last_generator_trace_id": link.last_generator_trace_id,
    }


async def _generate_turn(
    state: PipelineState, *, factory: AgentFactory, db: Database, adb: AsyncDatabase,
    speaker: Literal["customer", "agent"],
) -> dict[str, Any]:
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    if speaker == "customer":
        node_name, prompt_name = "customer_turn_generator", "phase2.customer_turn"
        inputs = _customer_inputs(state, slot)
    else:
        node_name, prompt_name = "agent_turn_generator", "phase2.agent_turn"
        inputs = _agent_inputs(state, slot)
    inputs["conversation_so_far"] = _render_history(state.current_dialogue_turns)
    inputs["turn_index"] = len(state.current_dialogue_turns) + 1
    # Prior consistency issues are surfaced to the agent turn on a re-roll.
    prior_issues = (
        [i.explanation for i in state.last_verdict.issues]
        if (speaker == "agent" and state.last_verdict is not None and not state.last_verdict.passed)
        else []
    )

    link = ParentLink()
    generator = factory.build_generator(
        name=node_name, prompt_name=prompt_name,
        output_schema_factory=lambda: DialogueTurnOutput,
    )
    traced = TracingAdapter(
        inner=generator, db=db, adb=adb, run_id=state.run_id, node_name=node_name,
        artifact_type="resolution", artifact_id=ticket_uid, role="generator", parent_link=link,
    )
    ctx = AgentContext(inputs={**inputs, "prior_issues": prior_issues}, retry_attempt=state.retry_attempt)
    result = await traced.invoke(ctx)
    assert isinstance(result, DialogueTurnOutput)
    result = result.model_copy(update={"speaker": speaker})
    turns = [*state.current_dialogue_turns, result]
    end_reason = None
    if result.done:
        end_reason = "customer_done" if speaker == "customer" else "agent_done"
    return {
        "current_dialogue_turns": turns,
        "dialogue_last_speaker": speaker,
        "dialogue_done": result.done,
        "dialogue_end_reason": end_reason,
        "last_generator_trace_id": link.last_generator_trace_id,
    }


async def generate_agent_turn_node(state, *, factory, db, adb, settings) -> dict[str, Any]:
    return await _generate_turn(state, factory=factory, db=db, adb=adb, speaker="agent")


async def generate_customer_turn_node(state, *, factory, db, adb, settings) -> dict[str, Any]:
    return await _generate_turn(state, factory=factory, db=db, adb=adb, speaker="customer")


async def _mark_cap_hit_node(state: PipelineState) -> dict[str, Any]:
    return {
        "dialogue_end_reason": "cap_hit",
        "last_quality_flag": "warning:turn_cap_hit",
    }


async def validate_conversation_node(
    state: PipelineState, *, factory: AgentFactory, db: Database, adb: AsyncDatabase, settings: AppSettings
) -> dict[str, Any]:
    slot = state.plan_slots[state.slot_index]
    ticket_uid = f"{state.run_id}:{slot.index:06d}"
    assert state.current_resolution_draft is not None
    end_reason = state.dialogue_end_reason or "agent_done"
    draft = _assemble_resolution(
        subject=state.current_resolution_draft.subject,
        body=state.current_resolution_draft.body,
        turns=state.current_dialogue_turns,
        end_reason=end_reason,
    )

    problem = {p.id: p for p in state.problems_committed}[slot.problem_id]
    inputs: dict[str, Any] = {
        "company_name": state.company.name,
        "customer_tone": slot.tone,
        "problem": {
            "title": problem.title,
            "complexity": problem.complexity,
            "symptoms": problem.symptoms,
            "root_cause": problem.root_cause,
            "resolution_hint": problem.resolution_hints.get(slot.ticket_type.value, ""),
        },
        "candidate": draft.model_dump(),
    }

    link = ParentLink(last_generator_trace_id=state.last_generator_trace_id)
    checker = factory.build_generator(
        name="conversation_consistency_check",
        prompt_name="phase2.conversation_consistency_check",
        output_schema_factory=lambda: ConsistencyVerdict,
    )
    traced = TracingAdapter(
        inner=checker, db=db, adb=adb, run_id=state.run_id,
        node_name="conversation_consistency_check", artifact_type="resolution",
        artifact_id=ticket_uid, role="checker", parent_link=link,
    )
    verdict = await traced.invoke(AgentContext(inputs=inputs, retry_attempt=state.retry_attempt))
    assert isinstance(verdict, ConsistencyVerdict)

    passed = verdict.status in ("pass", "pass_with_edits")
    edited = verdict.status == "pass_with_edits"
    final_draft = _apply_consistency_edits(draft, verdict) if passed else draft
    quality_flag = state.last_quality_flag
    if edited:
        quality_flag = "info:consistency_edited"
    return {
        "current_resolution_draft": final_draft,
        "current_dialogue_turns": final_draft.turns,
        "last_verdict": Verdict(checker="conversation_consistency_check", passed=passed, issues=[]),
        "last_quality_flag": quality_flag,
    }
```

Notes:
- `TracingAdapter` is given `role="checker"` for the consistency call but the underlying agent is a `Generator` bound to `ConsistencyVerdict` (the checker path needs a `Verdict`, which the consistency agent does not emit, so we use a generator + translate to a `Verdict` for routing). Confirm `TracingAdapter` accepts a `Generator` for `role="checker"`; if it strictly requires a `Checker`, set `role="generator"` here instead — the trace's `role` is cosmetic for routing and `parent_trace_id` linkage is preserved either way.
- `_mark_validation_skipped_node` is reused unchanged (already in the file).

- [ ] **Step 7: Update `commit_resolution_node` → `commit_dialogue_node`**

Rename `commit_resolution_node` to `commit_dialogue_node`. Inside it:
- Change `turns=[t.model_dump() for t in draft.turns]` — `draft.turns` are now `DialogueTurnOutput`, so `model_dump()` yields `{speaker, content, done, done_reason}` automatically. No `name` field.
- Add the per-slot dialogue resets to the returned dict:

```python
    return {
        "slot_index": state.slot_index + 1,
        "retry_attempt": 0,
        "current_resolution_draft": None,
        "current_dialogue_turns": [],
        "dialogue_last_speaker": None,
        "dialogue_done": False,
        "dialogue_end_reason": None,
        "last_verdict": None,
        "last_quality_flag": None,
        "last_generator_trace_id": None,
    }
```

The `ResolutionRecord(... resolved=draft.resolved ...)` line stays; `draft.resolved` is now correctly derived.

### 9c. Builder

- [ ] **Step 8: Rewrite `build_phase2_subgraph`**

Replace the node registrations and edges. Keep `build_allocation_plan`, `_bump_retry`, `_mark_exhausted`, `_mark_validation_skipped`. New wiring:

```python
    g.add_node("build_allocation_plan", partial(build_allocation_plan_node, db=db, adb=adb, settings=settings))
    g.add_node("generate_incoming_request", partial(generate_incoming_request_node, factory=factory, db=db, adb=adb, settings=settings))
    g.add_node("generate_agent_turn", partial(generate_agent_turn_node, factory=factory, db=db, adb=adb, settings=settings))
    g.add_node("generate_customer_turn", partial(generate_customer_turn_node, factory=factory, db=db, adb=adb, settings=settings))
    g.add_node("validate_conversation", partial(validate_conversation_node, factory=factory, db=db, adb=adb, settings=settings))
    g.add_node("commit_dialogue", partial(commit_dialogue_node, db=db, adb=adb))
    g.add_node("_bump_retry", _bump_retry_node)
    g.add_node("_mark_exhausted", _mark_exhausted_node)
    g.add_node("_mark_validation_skipped", _mark_validation_skipped_node)
    g.add_node("_mark_cap_hit", _mark_cap_hit_node)

    g.add_edge(START, "build_allocation_plan")
    g.add_conditional_edges("build_allocation_plan", _route_after_build_allocation_plan,
        {"generate": "generate_incoming_request", "done": END})
    g.add_edge("generate_incoming_request", "generate_agent_turn")
    g.add_conditional_edges("generate_agent_turn", _route_after_agent_turn,
        {"customer": "generate_customer_turn", "consistency": "_route_consistency_or_skip", "cap": "_mark_cap_hit"})
    g.add_conditional_edges("generate_customer_turn", _route_after_customer_turn,
        {"agent": "generate_agent_turn", "consistency": "_route_consistency_or_skip", "cap": "_mark_cap_hit"})
    g.add_edge("_mark_cap_hit", "_route_consistency_or_skip")
```

LangGraph conditional targets must be node names, so model "consistency or skip" as a tiny pass-through node plus a conditional edge. Add a no-op node and router:

```python
async def _converge_node(state: PipelineState) -> dict[str, Any]:
    return {}


def _route_validation_enabled(state: PipelineState) -> Literal["validate", "skip"]:
    return "validate" if state.validation_enabled else "skip"
```

Register and wire:

```python
    g.add_node("_route_consistency_or_skip", _converge_node)
    g.add_conditional_edges("_route_consistency_or_skip", _route_validation_enabled,
        {"validate": "validate_conversation", "skip": "_mark_validation_skipped"})
    g.add_edge("_mark_validation_skipped", "commit_dialogue")
    g.add_conditional_edges("validate_conversation", _route_after_validate_conversation,
        {"pass": "commit_dialogue", "retry": "_bump_retry", "exhausted": "_mark_exhausted"})
    g.add_edge("_bump_retry", "generate_incoming_request")
    g.add_edge("_mark_exhausted", "commit_dialogue")
    g.add_conditional_edges("commit_dialogue", _route_after_commit_resolution,
        {"next": "generate_incoming_request", "done": END})
```

- [ ] **Step 9: Update the builder-shape assertions in `tests/unit/test_phase2_graph.py`**

If the file asserts specific node names exist in the compiled graph, update them to the new node set: `build_allocation_plan`, `generate_incoming_request`, `generate_agent_turn`, `generate_customer_turn`, `validate_conversation`, `commit_dialogue`, `_route_consistency_or_skip`, `_mark_cap_hit`, `_mark_validation_skipped`, `_bump_retry`, `_mark_exhausted`. (Inspect the compiled graph via `g.get_graph().nodes` as the existing test does.)

- [ ] **Step 10: Run the full Phase 2 unit test file**

Run: `uv run pytest tests/unit/test_phase2_graph.py -v`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/csfd/graph/phase2_graph.py src/csfd/graph/pipeline_graph.py tests/unit/test_phase2_graph.py
git commit -m "feat(phase2): turn-based dialogue subgraph (nodes, routers, builder)"
```

---

## Task 10: Repair existing tests + rendering stub that reference removed names

**Files:**
- Modify: `src/csfd/graph/rendering.py` (lines 44, 71-72)
- Modify: `tests/unit/test_phase1_graph.py` (lines 78, 105-106)
- Modify: `tests/unit/test_pipeline_graph.py` (lines 68, 95-96)
- Modify: `tests/unit/test_prompt_render_smoke.py` (lines 104-200 — the resolution_generator / resolution_combined_check sections)
- Modify: `tests/integration/test_pipeline_retry.py` (lines 108, 152)
- Modify: `tests/integration/test_phase1_dedup.py` (lines 121, 149)
- Modify: `tests/integration/test_pipeline_deterministic.py` (line 24 import + usages)

- [ ] **Step 1: Fix `src/csfd/graph/rendering.py`**

Line 44: remove `turns_per_type={...}`, add `dialogue=DialogueConfig(turn_cap=20)` (import `DialogueConfig`). Lines 71-72: replace the two old prompt keys with the four new ones (`phase2.incoming_request`, `phase2.customer_turn`, `phase2.agent_turn`, `phase2.conversation_consistency_check`).

- [ ] **Step 2: Fix `test_phase1_graph.py` and `test_pipeline_graph.py`**

In each: remove `turns_per_type={...}` from the `TicketsConfig(...)` construction, add `dialogue=DialogueConfig(turn_cap=20)` (import it). Replace the two old prompt-handle keys with the four new ones.

- [ ] **Step 3: Rewrite the Phase 2 sections of `test_prompt_render_smoke.py`**

Delete the `resolution_generator` and `resolution_combined_check` test functions (they assert on `inputs.target_turn_count`, ticket-type branches, rule ids that no longer exist). Replace with minimal render-smoke tests for the new prompts:

```python
def test_incoming_request_renders(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.incoming_request")
    out = handle.render(inputs={
        "company_name": "Acme", "customer_name": "C-1", "customer_tier": "standard",
        "customer_tone": "neutral", "ticket_type": "l1", "category": "cooling",
        "symptoms": ["unit power-cycles"], "customer_impact": "degraded",
    })
    assert "unit power-cycles" in out


def test_agent_turn_renders_root_cause(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.agent_turn")
    out = handle.render(inputs={
        "company_name": "Acme", "agent_name": "A-1", "ticket_type": "l2",
        "problem": {"title": "T", "summary": "S", "background": "B", "category": "c",
                    "fault_domain": "software", "root_cause": ["bad config"], "resolution_hint": "reset"},
        "conversation_so_far": [{"speaker": "customer", "content": "help"}],
        "turn_index": 2, "turn_cap": 20,
    }, prior_issues=[])
    assert "bad config" in out


def test_consistency_check_renders(registry: PromptRegistry) -> None:
    handle = registry.get("phase2.conversation_consistency_check")
    out = handle.render(inputs={
        "company_name": "Acme", "customer_tone": "neutral",
        "problem": {"title": "T", "complexity": "simple", "symptoms": ["x"],
                    "root_cause": ["y"], "resolution_hint": "z"},
        "candidate": {"subject": "s", "body": "b", "end_reason": "agent_done",
                      "turns": [{"speaker": "customer", "content": "b", "done": False, "done_reason": None}]},
    })
    assert "simple" in out
```

Check the exact `handle.render(...)` signature used elsewhere in this file (it may take `inputs=` plus extra kwargs like `prior_verdicts`/`prior_issues`); match it.

- [ ] **Step 4: Fix integration tests `test_pipeline_retry.py`, `test_phase1_dedup.py`**

Remove `turns_per_type={...}`, add `dialogue=DialogueConfig(turn_cap=20)`. At line ~152 / ~149 these tests reference `"resolution_generator"` in a fake-LLM canned-response mapping or trace-node assertion — update to the new node names. For these phase-1-focused tests, Phase 2 may run with `tickets.total=0` or a trivial slot; if they assert a Phase 2 trace `node_name`, change `"resolution_generator"` to `"incoming_request_generator"`. Read the assertion in context before editing.

- [ ] **Step 5: Fix `test_pipeline_deterministic.py`**

Line 24 imports `ResolutionTurnOutput` — change to `DialogueTurnOutput`. Update any construction of resolution turns to the new shape (`speaker`, `content`, `done`, `done_reason`; no `name`). Update canned fake-LLM structured outputs: the deterministic pipeline test must now provide canned `IncomingRequestOutput`, `DialogueTurnOutput` (use `structured_seq` so the loop can terminate — provide a customer/agent turn sequence whose last turn has `done=True`), and `ConsistencyVerdict(status="pass")`. Set `tickets.dialogue=DialogueConfig(turn_cap=20)`. This is the most involved fix — model it on the new happy-path integration test (Task 11) and keep the deterministic seed/counts the test already pins.

- [ ] **Step 6: Delete the two superseded prompts**

```bash
git rm prompts/phase2/resolution_generator.md.j2 prompts/phase2/resolution_combined_check.md.j2
```

Then confirm nothing still references them:
Run: `grep -rn "resolution_generator\|resolution_combined_check\|turns_per_type\|ResolutionTurnOutput" src/ tests/ config/`
Expected: no output.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/csfd/graph/rendering.py tests/ prompts/phase2/
git commit -m "test: migrate existing tests to turn-based dialogue schema + nodes"
```

---

## Task 11: Integration test — happy path

**Files:**
- Create: `tests/integration/test_phase2_dialogue_happy_path.py`

Model the harness on `tests/integration/test_pipeline_deterministic.py` (DB setup, factory wiring, `build_phase2_subgraph`, `PipelineState` seeding with one committed problem and a 1-slot plan).

- [ ] **Step 1: Write the test**

```python
"""Integration: a scripted dialogue ends when the agent flags done; one slot commits."""

from __future__ import annotations

# ... imports mirroring test_pipeline_deterministic.py: Database, AsyncDatabase,
# AgentFactory, FakeChatModel, PromptRegistry/PromptHandle stubs, build_phase2_subgraph,
# PipelineState, ProblemRecord, _PlanSlot ...

from csfd.pipeline import (
    ConsistencyVerdict, DialogueTurnOutput, IncomingRequestOutput,
)


def _problem() -> "ProblemRecord":
    # symptoms-only customer view + root_cause for the agent
    return ProblemRecord(
        id="p001", run_id="r", title="Power cycling", summary="PSU brownout",
        background="...", category="cooling", complexity="simple",
        resolution_hints={"l1": "reseat power"}, quality_flag=None,
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        symptoms=["unit power-cycles every 20 min"], root_cause=["loose PSU connector"],
        fault_domain="hardware", customer_impact="degraded", tags=[],
    )


def _factory_with_script() -> AgentFactory:
    fake = FakeChatModel(
        structured={
            IncomingRequestOutput: IncomingRequestOutput(subject="Power", body="It power-cycles."),
            ConsistencyVerdict: ConsistencyVerdict(status="pass"),
        },
        structured_seq={
            DialogueTurnOutput: [
                DialogueTurnOutput(speaker="agent", content="Reseat the connector.", done=False),
                DialogueTurnOutput(speaker="customer", content="That fixed it, thanks!",
                                   done=True, done_reason="customer_satisfied"),
            ],
        },
    )
    # ... wire registry handles for the four phase2 prompts to a stub handle,
    # llm_builder returning `fake`, agent_configs for generator+combined_checker ...
    return factory


async def test_dialogue_happy_path_commits_resolved(tmp_path) -> None:
    # build db, run migrations, seed state with one problem + run build_phase2_subgraph,
    # ainvoke it once.
    final = await graph.ainvoke(state)

    # 1 incoming request, 1 resolution persisted
    res_rows = ResolutionRepo(db).list_for_run("r")  # or direct SQL
    assert len(res_rows) == 1
    row = res_rows[0]
    assert row.turn_count == 3                      # opening + agent + customer
    assert row.resolved is True
    assert row.turns[0]["speaker"] == "customer"
    assert row.turns[-1]["done"] is True

    # agent_traces: 1 incoming + 2 turns + 1 consistency = 4 for the slot
    trace_count = ...  # SELECT COUNT(*) FROM agent_traces WHERE run_id='r'
    assert trace_count == 4
```

Fill in the harness from the deterministic test. Assert: `turn_count == 3`, `resolved is True`, last turn `done is True`, and `agent_traces` row count == 4 for the slot.

- [ ] **Step 2: Run, verify pass**

Run: `uv run pytest tests/integration/test_phase2_dialogue_happy_path.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase2_dialogue_happy_path.py
git commit -m "test(phase2): integration happy-path dialogue"
```

---

## Task 12: Integration test — cap hit

**Files:**
- Create: `tests/integration/test_phase2_dialogue_cap_hit.py`

- [ ] **Step 1: Write the test**

Reuse the harness from Task 11 but set `dialogue_turn_cap` low (e.g. 4) via the stub settings (`DialogueConfig(turn_cap=4)`), and provide `structured_seq[DialogueTurnOutput]` of turns that NEVER set `done=True` (supply enough entries to exceed the cap — e.g. 10 identical `done=False` turns). Assert:

```python
    assert row.turn_count == 4               # cap reached
    assert row.resolved is False
    assert row.quality_flag == "warning:turn_cap_hit"
```

Note: `build_allocation_plan_node` seeds `dialogue_turn_cap` from `settings.tickets.dialogue.turn_cap`, so set the stub settings' cap to 4 rather than overriding state directly.

- [ ] **Step 2: Run, verify pass**

Run: `uv run pytest tests/integration/test_phase2_dialogue_cap_hit.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase2_dialogue_cap_hit.py
git commit -m "test(phase2): integration turn-cap hit path"
```

---

## Task 13: Integration test — consistency retry

**Files:**
- Create: `tests/integration/test_phase2_dialogue_consistency_retry.py`

- [ ] **Step 1: Write the test**

Harness from Task 11, with `validation_enabled=True` and `max_retries=1`. Make the consistency agent fail once then pass: provide `structured_seq[ConsistencyVerdict] = [ConsistencyVerdict(status="fail", issues=["bad"]), ConsistencyVerdict(status="pass")]`. Because a fail re-rolls the whole conversation from `generate_incoming_request`, the `structured_seq[DialogueTurnOutput]` queue must contain TWO full conversations' worth of turns (first roll + second roll). Provide both, each ending with a `done=True` turn. Assert:

```python
    assert len(res_rows) == 1               # only the final, passing conversation commits
    assert row.resolved is True
    # consistency was called twice (fail, then pass)
    consistency_traces = ...  # COUNT WHERE node_name='conversation_consistency_check'
    assert consistency_traces == 2
```

- [ ] **Step 2: Run, verify pass**

Run: `uv run pytest tests/integration/test_phase2_dialogue_consistency_retry.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase2_dialogue_consistency_retry.py
git commit -m "test(phase2): integration consistency-fail re-roll"
```

---

## Task 14: Integration test — consistency edit

**Files:**
- Create: `tests/integration/test_phase2_dialogue_consistency_edit.py`

- [ ] **Step 1: Write the test**

Harness from Task 11. Script a 3-turn conversation, then a single `ConsistencyVerdict`:

```python
edited = [
    DialogueTurnOutput(speaker="customer", content="It power-cycles.", done=False),
    DialogueTurnOutput(speaker="agent", content="Reseat the connector please.", done=False),
    DialogueTurnOutput(speaker="customer", content="Done — resolved, thank you.",
                       done=True, done_reason="customer_satisfied"),
]
verdict = ConsistencyVerdict(
    status="pass_with_edits", issues=["smoothed wording"],
    edited_subject="Power cycling resolved", edited_body="It power-cycles.",
    edited_turns=edited,
)
```

Assert the persisted resolution reflects the edits and the flag:

```python
    assert row.turns[1]["content"] == "Reseat the connector please."
    assert row.quality_flag == "info:consistency_edited"
    assert row.resolved is True
```

- [ ] **Step 2: Run, verify pass**

Run: `uv run pytest tests/integration/test_phase2_dialogue_consistency_edit.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase2_dialogue_consistency_edit.py
git commit -m "test(phase2): integration consistency pass-with-edits"
```

---

## Task 15: Update README + final full-suite + lint

**Files:**
- Modify: `README.md` (tickets / Phase 2 sections)

- [ ] **Step 1: Edit README**

In the `tickets` config description, remove the `turns_per_type` bullet and add a `dialogue.turn_cap` bullet. In the Phase 2 description ("Phase 2 — Resolution per slot"), replace "for each slot, one LLM call produces the standalone incoming request plus the full multi-turn resolution" with a sentence describing the turn-based dual-agent loop: customer and service agents alternate, each turn is one LLM call, conversation length is emergent (simple problems 2–3 turns, complex more), capped at `dialogue.turn_cap`, then a consistency agent reviews/edits the transcript. Note the per-ticket LLM-call increase (~4–10 turn calls + 1 consistency vs. the old 1 generator + 1 checker).

- [ ] **Step 2: Run the full suite + linters**

Run:
```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```
Expected: all green. Fix any type errors (notably: `Literal` annotations on the new state fields and node returns, and `model_dump()` on `DialogueTurnOutput`).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): document turn-based dual-agent Phase 2"
```

---

## Self-Review Notes (verification the plan covers the spec)

- Info asymmetry (spec §Information asymmetry): enforced in `_customer_inputs` / `_agent_inputs` (Task 9) — customer gets `symptoms`/`customer_impact`/`customer_tone`, agent gets `root_cause`/`background`/`summary`/`resolution_hint`, neither gets `complexity`; consistency gets both + complexity. ✅
- Termination (whichever speaker flags done; 20-turn cap): routers + `_mark_cap_hit_node` (Task 9), cap from `dialogue.turn_cap` (Tasks 4–5). ✅
- Consistency at end, edit-in-place, `info:consistency_edited`: `validate_conversation_node` + `_apply_consistency_edits` (Tasks 2, 9). ✅
- Miscommunication via persona prompts: `customer_turn.md.j2` / `agent_turn.md.j2` (Task 8). ✅
- Simple problems short (2–3 turns): length guidance in both turn prompts + complexity-aware consistency check (Task 8). ✅
- Schemas + `resolved` derivation + turns_json shape (no migration): Tasks 1, 9 (`commit_dialogue_node` dumps `DialogueTurnOutput`). ✅
- Config: `turns_per_type` removed, `dialogue.turn_cap` added (Tasks 4–5); run snapshot picks it up via `settings.tickets.model_dump()` already in `init_run_node` — no extra change. ✅
- Trace node names + counts: Task 9 node_names; happy-path asserts 4 traces/slot (Task 11). ✅
- Old prompts deleted: Task 8 creates new; deletion of `resolution_generator.md.j2` / `resolution_combined_check.md.j2` happens in Task 10 cleanup — **add explicit `git rm` there**.
