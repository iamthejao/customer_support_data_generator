# Turn-Based Dual-Agent Ticket Resolution — Design

**Date:** 2026-05-27
**Status:** Approved for implementation planning
**Scope:** Phase 2 (resolution generation) only

## Motivation

Today's Phase 2 generates the entire customer-service conversation in a single LLM call driven by a fixed `turns_per_type` config. Two problems with this:

1. **Turn count is artificial.** A model asked to produce a 6-turn dialogue will pad even when the problem can be resolved in two turns, and will truncate when realistic resolution would take ten.
2. **Both speakers share information.** One prompt is asked to roleplay both customer and agent simultaneously, so the model knows the root cause when writing the customer's lines. The resulting conversations rarely show real information discovery or miscommunication — the "customer" already knows the answer.

This work replaces the single-shot generator with a two-agent dialogue loop in which conversation length is emergent and information asymmetry is enforced at the prompt-input boundary.

## Goals

- Conversation length emerges from the problem and the dialogue, not from config.
- Customer agent has only customer-observable information (symptoms, impact, persona). Service agent has root-cause information but does not see the customer's symptoms directly — it must read them from the conversation.
- Conversations show realistic ping-pong and occasional miscommunication, arising from persona-driven prompts (not injected synthetically).
- A consistency agent reviews the full transcript at the end and may edit it in place.
- Simple problems resolve in 2–3 turns; complex problems naturally take more.

## Non-goals

- Phase 1 problem generation is unchanged.
- The `incoming_requests` / `resolutions` / `lineage` / `agent_traces` table schemas are unchanged.
- No live human-in-the-loop dialogue. Both speakers are LLMs.
- No backwards compatibility with pre-change checkpoints.

## Architecture

### Per-slot dialogue loop

Replaces the existing `generate_resolution → validate_resolution → commit_resolution` cycle in `src/csfd/graph/phase2_graph.py`. The allocation plan, lineage pre-recording, retry-budget machinery, and outer per-slot iteration are unchanged.

```
build_allocation_plan
       │
       ▼
generate_incoming_request          ← 1 LLM call: customer's opening message (subject + body)
       │                             from symptoms + persona. Becomes turn 1 (customer).
       ▼
generate_agent_turn   ←─────┐      ← 1 LLM call: service agent response. Emits {content, done}.
       │                    │
       ▼                    │
_route_after_agent_turn     │
   │  │  │                  │
   │  │  └ cap ──────────┐  │
   │  └ done ──────────┐ │  │
   ▼                   │ │  │
generate_customer_turn │ │  │      ← 1 LLM call: customer response. Emits {content, done}.
   │                   │ │  │
   ▼                   │ │  │
_route_after_customer_turn │  │
   │  │  │             │ │  │
   │  │  └ cap ─────┐  │ │  │
   │  └ done ─────┐ │  │ │  │
   └ next ────────┼─┼──┼─┼──┘
                  ▼ ▼  ▼ ▼
              _mark_cap_hit (cap branch only)
                  │
                  ▼
        validate_conversation                ← LLM: consistency review on full draft.
                  │                            Returns pass | pass_with_edits | fail.
                  ▼
        _route_after_validate_conversation
        ┌────┬───────┴─────────┐
        │pass│ pass_with_edits │ fail
        ▼    ▼                 │
      apply consistency edits  │  (retries left? yes: _bump_retry → restart at
        (if any)               │   generate_incoming_request; no: _mark_exhausted)
                  │            │
                  ▼            ▼
              commit_dialogue ←─── _mark_exhausted
                  │
                  ▼
        _route_after_commit (next slot / END)
```

Key topology choices:

- Turn 1 is special: `generate_incoming_request` produces the email envelope (`subject`, `body`) and seeds `current_dialogue_turns[0]` as the customer's opening turn. The `incoming_requests` table semantics are preserved — the incoming request **is** turn 1.
- Alternation starts with the service agent after turn 1.
- A consistency failure throws away the whole conversation and re-rolls from turn 1. Partial re-rolls are not attempted; keeping multi-turn state internally consistent under partial edits is hard and not worth the complexity.
- Cap-hit conversations still commit (`resolved=false`, `quality_flag="warning:turn_cap_hit"`); they are not retries.

## Information asymmetry

The split is enforced at the prompt-input boundary. Each agent's prompt receives only the fields it is allowed to see; nothing in `PipelineState` per se prevents leakage, but no node passes forbidden fields into a prompt.

| Field | Customer agent | Service agent |
|---|---|---|
| `problem.symptoms_json` | ✅ | ❌ (must read from messages) |
| `problem.customer_impact` | ✅ | ❌ |
| `problem.title` | ❌ | ✅ |
| `problem.summary` | ❌ | ✅ |
| `problem.background` | ❌ | ✅ |
| `problem.category` | ✅ | ✅ |
| `problem.fault_domain` | ❌ | ✅ |
| `problem.root_cause_json` | ❌ | ✅ |
| `problem.resolution_hints[ticket_type]` | ❌ | ✅ |
| `problem.complexity` | ❌ | ❌ (consistency agent only) |
| `customer_tone` | ✅ | ❌ (must infer from messages) |
| `customer_tier` | ✅ | ✅ (slot context) |
| `ticket_type` | ✅ | ✅ |
| `conversation_so_far` | ✅ | ✅ |

The consistency agent sees everything from both sides plus `problem.complexity` so it can judge whether conversation length is appropriate to the problem.

## Schemas

### New Pydantic models (in `src/csfd/pipeline.py`)

```python
class DialogueTurnOutput(BaseModel):
    speaker: Literal["customer", "agent"]
    content: str
    done: bool                          # speaker thinks conversation is over after this turn
    done_reason: str | None = None      # short tag, e.g. "resolved" / "needs_followup"

class IncomingRequestOutput(BaseModel):
    subject: str
    body: str

class ConsistencyVerdict(BaseModel):
    status: Literal["pass", "pass_with_edits", "fail"]
    issues: list[str]
    edited_turns: list[DialogueTurnOutput] | None = None
    edited_subject: str | None = None
    edited_body: str | None = None
```

### `ResolutionOutput` evolves to

```python
class ResolutionOutput(BaseModel):
    subject: str
    body: str
    turns: list[DialogueTurnOutput]
    resolved: bool                                            # derived, see below
    end_reason: Literal["customer_done", "agent_done", "cap_hit"]
```

`ResolutionTurnOutput` is removed.

### `PipelineState` additions

```python
current_dialogue_turns: list[DialogueTurnOutput] = []
dialogue_last_speaker: Literal["customer", "agent"] | None = None
dialogue_done: bool = False
dialogue_end_reason: Literal["customer_done", "agent_done", "cap_hit"] | None = None
```

Existing `current_resolution_draft` is populated only at the consistency step from the accumulated turns, so the commit node's shape is unchanged.

### Resolved-flag derivation

A pure helper:

```python
RESOLVED_DONE_REASONS = {"resolved", "customer_satisfied", "issue_fixed", "closed"}

def _derive_resolved(end_reason: str, last_done_reason: str | None) -> bool:
    if end_reason == "cap_hit":
        return False
    return last_done_reason in RESOLVED_DONE_REASONS
```

`done_reason` vocabulary is described in prompts as a soft suggestion list. Free-text values are allowed; only the listed tags map to `resolved=True`.

## Prompts

Four new prompts in `prompts/phase2/`. The existing `resolution_generator.md.j2` and `resolution_combined_check.md.j2` are deleted.

### `phase2/incoming_request.md.j2` — customer's opening message
Inputs: `company_name`, `customer_name`, `customer_tier`, `customer_tone`, `ticket_type`, `problem.symptoms`, `problem.customer_impact`, `problem.category`.
Output schema: `IncomingRequestOutput`.
Persona instruction: roleplay as someone who only knows what they're experiencing. Write in `customer_tone`. Be partial / vague when realistic. Do not state diagnostic information.

### `phase2/customer_turn.md.j2` — customer follow-up turns
Inputs: same persona block, plus `conversation_so_far`, `turn_index`, `turn_cap`.
Output schema: `DialogueTurnOutput` with `speaker="customer"`.
Behavior: keep `customer_tone` consistent; answer agent questions from your symptoms only; occasionally misinterpret or ask for clarification when an agent's question is ambiguous; decide `done=true` when satisfied, frustrated-and-leaving, or told the issue is resolved.
**Length guidance:** *"Short conversations are correct for simple problems. If the agent's solution clearly addresses what you described, acknowledge it and set `done=true`. Don't invent extra questions."*

### `phase2/agent_turn.md.j2` — service-agent turns
Inputs: `company_name`, `agent_name`, `ticket_type`, `problem.title`, `problem.summary`, `problem.background`, `problem.category`, `problem.fault_domain`, `problem.root_cause`, `problem.resolution_hints[ticket_type]`, `conversation_so_far`, `turn_index`, `turn_cap`.
Output schema: `DialogueTurnOutput` with `speaker="agent"`.
Behavior: ask clarifying questions when symptoms in messages are ambiguous; progressively reveal diagnostic reasoning; guide toward the resolution hint; decide `done=true` when the issue is resolved, the customer is satisfied, or escalation/closure is appropriate.
**Length guidance:** *"Short conversations are correct for simple problems. If the symptoms are clear enough to answer in one turn, do so. Do not pad with unnecessary clarifying questions when the customer's description is already sufficient."*

### `phase2/conversation_consistency_check.md.j2` — post-hoc review
Inputs: full assembled `ResolutionOutput` (subject, body, turns, end_reason), plus the **full problem** (both customer-view and root-cause fields) and `problem.complexity` for cross-checking.
Output schema: `ConsistencyVerdict`.
Checks:
- (a) The customer never uses root-cause language they could not know.
- (b) The agent's diagnostic path is plausible given the symptoms they were given.
- (c) Information ping-pong is appropriate to `problem.complexity` — simple/clear symptoms can resolve in 2–3 turns; medium and complex problems should show progressive discovery.
- (d) Tone is consistent with `customer_tone`.
- (e) No contradictions between turns.
- (f) The `resolved` flag (and `end_reason`) matches what the conversation actually achieved.

## LangGraph nodes & routers

New nodes in `src/csfd/graph/phase2_graph.py`:

- `generate_incoming_request_node` — LLM call for customer's opening message. Seeds `current_dialogue_turns[0]` with `{speaker: "customer", content: body, done: False, done_reason: None}` (the opening turn never ends the conversation). Sets `dialogue_last_speaker="customer"`.
- `generate_agent_turn_node` — appends one service-agent turn to `current_dialogue_turns`. Renders `conversation_so_far` from accumulated turns. Sets `dialogue_last_speaker="agent"` and `dialogue_done` from the returned turn.
- `generate_customer_turn_node` — mirror of the above for the customer side, using the customer-view inputs.
- `validate_conversation_node` — assembles a `ResolutionOutput` from accumulated state, runs the consistency agent, persists the verdict. On `pass_with_edits`, updates `current_resolution_draft` + `current_dialogue_turns` from the edited fields.
- `commit_dialogue_node` — same shape as today's `commit_resolution_node`. Persists the final draft, backfills lineage, clears per-slot dialogue state.

Routers:

```python
def _route_after_agent_turn(state) -> Literal["customer", "consistency", "cap"]:
    if state.dialogue_done: return "consistency"
    if len(state.current_dialogue_turns) >= DIALOGUE_TURN_CAP: return "cap"
    return "customer"

def _route_after_customer_turn(state) -> Literal["agent", "consistency", "cap"]:
    if state.dialogue_done: return "consistency"
    if len(state.current_dialogue_turns) >= DIALOGUE_TURN_CAP: return "cap"
    return "agent"

def _route_after_validate_conversation(state) -> Literal["pass", "retry", "exhausted"]:
    # same shape as today's _route_after_validate_resolution
```

Preserved from today's graph: `build_allocation_plan_node`, `_bump_retry_node`, `_mark_exhausted_node`, `_route_after_build_allocation_plan`, `_route_after_commit_resolution`. When `validation_enabled=False`, the post-loop branch goes through `_mark_validation_skipped` → `commit_dialogue` (mirrors today's pattern).

Removed: `generate_resolution_node`, `validate_resolution_node`, `_route_after_generate_resolution`, `_route_after_validate_resolution`.

## Persistence

### `resolutions.turns_json` blob shape

Per-turn JSON changes from `{speaker, name, content}` to `{speaker, content, done, done_reason}`. `name` is dropped (it was always derived from slot context and reconstructable from lineage). `turns_json` is opaque JSON; no SQL migration is required.

### `resolutions.resolved`

Derived via `_derive_resolved(end_reason, last_done_reason)` (see Schemas). Unit-tested.

### Quality-flag vocabulary

| Flag | When |
|---|---|
| `"warning:turn_cap_hit"` | Conversation hit `DIALOGUE_TURN_CAP` without a `done` signal. |
| `"warning:retries_exhausted"` | Consistency failed all retries; committed anyway. (Existing.) |
| `"warning:validation_skipped"` | `validation_enabled=False`. (Existing.) |
| `"info:consistency_edited"` | Consistency returned `pass_with_edits`. (New.) |

### `incoming_requests` / `lineage` tables

Unchanged. `subject` and `body` come from `generate_incoming_request_node`'s output. Lineage is pre-recorded at allocation time and backfilled at commit, as today.

### `agent_traces`

Unchanged at the schema level. Many more rows per slot than today (typically 4–10 turn calls + 1 consistency call vs. today's 1 + 1). New `node_name` values: `incoming_request_generator`, `customer_turn_generator`, `agent_turn_generator`, `conversation_consistency_check`. `artifact_type` stays `"resolution"`, `artifact_id` stays `ticket_uid`. `parent_trace_id` is set only on the consistency call (linking to the last turn-generator trace); per-turn calls have no parent link.

## Config changes

### `config/default.yaml`

```yaml
tickets:
  total: ...
  type_proportions: ...
  tier_proportions: ...
  tone_proportions_per_type: ...
  # REMOVED: turns_per_type
  dialogue:
    turn_cap: 20                   # hard safety cap; rarely hit in practice
  assignment_strategy: ...
```

`AppSettings` loses the `tickets.turns_per_type` field. Profile overlays under `config/profiles/*.yaml` get the same removal. `validation` section is unchanged; the `combined_checker` agent bucket is reused for the conversation consistency call.

### `DIALOGUE_TURN_CAP`

Read from `settings.tickets.dialogue.turn_cap` at Phase 2 subgraph build time and closed over by the routers (`_route_after_agent_turn`, `_route_after_customer_turn`). No module-level constant; the cap is part of the run's config snapshot for provenance.

### Backwards compatibility

None. This is a destructive replacement:
- Prompts and graph nodes are not dual-mode.
- A run started with this change cannot resume from a pre-change checkpoint; the `PipelineState` shape differs.
- Old `resolutions.turns_json` rows from prior runs remain readable (JSON tolerates extra/missing fields), but new rows have the new shape.

This is consistent with how Phase 1 dedup was rolled out.

## Testing strategy

### Unit tests (under `tests/unit/`)

- `test_dialogue_state.py` — `_derive_resolved` truth table; `DialogueTurnOutput` / `ResolutionOutput` schema validation; transcript-assembly helper.
- `test_phase2_routers.py` — `_route_after_agent_turn`, `_route_after_customer_turn`, `_route_after_validate_conversation` on hand-built `PipelineState` instances covering all branches (done → consistency, not-done < cap → next speaker, cap → consistency, validate pass / retry / exhausted).
- `test_consistency_verdict.py` — `ConsistencyVerdict` schema; edit application (pass_with_edits → assembled draft).

### Integration tests (under `tests/integration/`)

Using the existing fake-LLM fixture pattern (response stubbed per `prompt_name`):

- `test_phase2_dialogue_happy_path.py` — scripted turns end with agent flagging done. Assert (a) conversation ends correctly, (b) `resolutions` row has correct `turn_count` and `resolved`, (c) `agent_traces` has N+1 rows per slot.
- `test_phase2_dialogue_cap_hit.py` — stub agents never flag done. Assert cap fires at `DIALOGUE_TURN_CAP`, `resolved=False`, `quality_flag="warning:turn_cap_hit"`.
- `test_phase2_dialogue_consistency_retry.py` — first consistency verdict fails, second passes. Assert full re-roll happens and second conversation commits.
- `test_phase2_dialogue_consistency_edit.py` — consistency returns `pass_with_edits`. Assert edited turns are what's persisted, `quality_flag="info:consistency_edited"`.

## Cost note

Per-slot LLM calls go from ~2 (generator + checker) to typically ~4–10 (one per turn + consistency). For a 1000-ticket run with a simple-majority complexity mix, expected total LLM calls roughly 4000–6000, vs today's ~2000. Worth surfacing in the budget pre-check and the README's tickets section.
