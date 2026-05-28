# Diversity Control & Reproducibility Contract — Design

**Date:** 2026-05-28
**Status:** Approved (pending spec review)
**Scope:** Items 1 and 4 from `IMPROVEMENTS.md` (Codex approach review).

## Context

An approach-level review (`IMPROVEMENTS.md`) flagged five themes. This spec covers
the two the user prioritized:

- **Item 1 — Reproducibility contract** (doc-only): the README determinism language
  understates LLM/model variance and the "Planned improvements" list is stale.
- **Item 4 — Diversity control** (code): the allocator collapses each ticket type's
  problems into a single complexity bucket, and Phase 1 dedup only embeds
  `title + summary`.

The user narrowed Item 4 to two changes: (4a) use the full complexity range per
type, and (4b) deepen the dedup embedding text. Per-problem usage caps and
fault_domain/category balancing were explicitly deferred.

The two workstreams are independent and can land as separate PRs.

---

## Item 1 — Reproducibility contract (doc-only, `README.md`)

No code changes. Tighten the existing `## Reproducibility` section (currently
~lines 200–214) and fix a stale list.

1. **Add an explicit reproducibility contract** at the top of the section, stating
   plainly what is and is not reproducible:
   - **Reproducible** given identical config + seed + fresh DB: the allocation plan
     and slot-by-slot lineage indices `(slot_index, problem_index_within_run,
     ticket_type, customer_tier, customer_tone, customer_name)`, plus all provenance
     stamps (config snapshot, prompt_id, model_id, git_sha).
   - **Not reproducible**: generated text. Agent temperatures are nonzero and
     provider models drift, so problem/resolution *content* is not bitwise
     reproducible. This promotes the currently-buried "the only source of additional
     variation are the LLM responses themselves" line into a clear contract.

2. **Document the model-revision gap**: `agent_traces.model_id` records the model
   *name* but not the provider snapshot/revision, so a provider-side model upgrade
   under the same name silently changes outputs without changing the recorded
   provenance. State this as a known limitation and suggest pinning model snapshot
   identifiers in config for stricter provenance. (No code change in this spec.)

3. **Fix the stale "Planned improvements" list**: item #3 "Turn-based ticket
   creation" is already implemented (it is the current Phase 2 dialogue design — see
   `2026-05-27-turn-based-dialogue-design.md`). Remove it. Item #2 (noise agent) and
   item #1 (prompt optimization) remain valid.

**Acceptance:** the Reproducibility section opens with a clear reproducible /
not-reproducible contract; the model-revision limitation is documented; the
Planned improvements list no longer advertises shipped work.

---

## Item 4a — Allocator uses full complexity range

**File:** `src/csfd/allocator.py` (`_assign_complexity_weighted`).

### Current behavior (the bug)

For each ticket type, the function walks `complexity_preference` then a global
fallback order, picks the **first non-empty bucket**, and round-robins problem ids
**only within that one bucket**. Consequence: an L2 type preferring
`(medium, complex)` that finds any `medium` problems will *never* use `complex`
problems — a diversity collapse.

### New behavior

Distribute a type's `n` slots across **all non-empty matching complexity buckets**
in its preference tuple, **weighted by preference rank** (first preferred gets the
larger share). Within each chosen bucket, round-robin problem ids as today.

- **Weights:** hardcoded rank-decay weights (no config knob — YAGNI). For the common
  cases:
  - 1 preferred bucket → `[1.0]`
  - 2 preferred buckets → `[0.6, 0.4]`
  - 3 preferred buckets → `[0.5, 0.3, 0.2]`

  Implement as a small deterministic rule (e.g. a fixed lookup keyed by the number of
  *non-empty* preferred buckets) rather than a computed decay, so the values are
  explicit and testable.
- **Integer split:** convert weights → per-bucket integer slot counts via the
  existing `largest_remainder` helper (`csfd.utils.rng`). This keeps the strategy
  deterministic **with no seed** — preserving the current "complexity_weighted is
  reproducible without a seed" guarantee.
- **Within-bucket assignment:** round-robin the sorted-by-id problem list for that
  bucket, exactly as today, for the count that bucket was allocated.
- **Fallback:**
  - If only some preferred buckets are empty → compute weights over the **non-empty
    preferred buckets only** (renormalized), so empty buckets contribute nothing.
  - If **all** preferred buckets are empty → fall back to the first non-empty bucket
    in global order `simple → medium → complex` and round-robin within it (current
    behavior, unchanged).

### Determinism / compatibility note

This changes the slot→problem assignment for `complexity_weighted` runs, so plans
generated before this change will differ after it. That is acceptable (pre-release;
the determinism guarantee is about *re-running the same code+config*, not
cross-version stability). README line ~203 describing the strategy is updated to
reflect rank-weighted multi-bucket assignment.

### Acceptance

- A type with two non-empty preferred buckets draws problems from **both**, in the
  weighted ratio (modulo `largest_remainder` rounding).
- Two runs with identical config (no seed) produce identical assignments.
- Empty-bucket fallback paths behave as specified.

---

## Item 4b — Deeper Phase 1 dedup embedding text

**Files:** `src/csfd/embeddings/text.py`, `config/profiles/local-only.yaml`,
`config/profiles/mixed.yaml`.

### Change

- Add a new `TextTemplate` literal value: `title_summary_symptoms_root_cause`.
- Extend the `_ProblemLike` Protocol with `symptoms: list[str]` and
  `root_cause: list[str]`. Both fields already exist on `ProblemBrainstormOutput`
  (`src/csfd/pipeline.py`, the commit-time candidate) and on `ProblemRecord`
  (`src/csfd/storage/repository.py`), so no schema work is required.
- New template body joins title, summary, and the two lists. Lists are rendered
  deterministically (joined on newlines, in stored order) so the embedded string is
  stable:

  ```
  {title}

  {summary}

  Symptoms:
  - {symptom_1}
  - {symptom_2}

  Root cause:
  - {cause_1}
  ```

- `text.py` remains the single source of truth, so commit-time embedding and any
  later re-embedding derive from identical strings automatically.

### Default

- Set `embedding.text_template: title_summary_symptoms_root_cause` in
  `config/profiles/local-only.yaml` and `config/profiles/mixed.yaml` (the profiles
  that enable embedding).
- Leave `config/default.yaml` on `title_summary` (other profiles don't run
  embedding, so the default value is inert there).

### Acceptance

- The new template renders the expected string for a problem with symptoms and
  root_cause populated, and degrades gracefully (omits empty list sections) when
  they are absent.
- `local-only` and `mixed` profiles resolve to the new template.

---

## Testing

- **Allocator** (`tests/`): unit tests asserting (a) both preferred buckets are used
  with the weighted ratio, (b) determinism across two builds with identical config
  and no seed, (c) renormalization when one preferred bucket is empty, (d) global
  fallback when all preferred buckets are empty.
- **Dedup text**: unit test the new template string for a populated problem and for
  one with empty symptoms/root_cause.
- Existing allocator tests for `uniform` and for proportion rounding remain green.

## Out of scope

Per-problem usage caps, fault_domain/category balancing, hard non-LLM validators,
budget enforcement, and the "accepted vs benchmark-grade" split — all deferred
(see `IMPROVEMENTS.md`).
