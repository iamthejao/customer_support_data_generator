# Diversity Control & Reproducibility Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the allocator draw from a ticket type's full preferred complexity range, deepen Phase 1 dedup with a symptoms+root_cause embedding template, and tighten the README reproducibility contract.

**Architecture:** Three independent changes. (1) Rewrite `_assign_complexity_weighted` in `allocator.py` to split slots across all non-empty preferred complexity buckets via rank-decay weights + `largest_remainder` (deterministic, no seed). (2) Add a new `TextTemplate` value and extend the embed-text protocol in `embeddings/text.py`, mirror the Literal in `settings.py`, and default it on in the two embedding profiles. (3) Edit `README.md` prose only.

**Tech Stack:** Python 3.12, pydantic, pytest, mypy (strict), ruff. Run via `uv run`.

---

### Task 1: Allocator uses full complexity range

**Files:**
- Modify: `src/csfd/allocator.py` (`_assign_complexity_weighted`, ~lines 174-217)
- Test: `tests/unit/test_allocator.py`

- [ ] **Step 1: Write the failing tests**

Append these three tests to `tests/unit/test_allocator.py` (the existing `_problems`, `TYPE_PROPS`, etc. helpers are already defined at the top of that file):

```python
def test_complexity_weighted_uses_both_preferred_buckets() -> None:
    # L2 prefers (medium, complex); both populated -> rank weights 0.6/0.4 over 10 slots.
    problems = _problems({ProblemComplexity.MEDIUM: 5, ProblemComplexity.COMPLEX: 5})
    plan = build_allocation_plan(
        total=10,
        type_proportions={"docs_request": 0.0, "l1": 0.0, "l2": 1.0, "l3": 0.0},
        tier_proportions={"standard": 1.0},
        tone_proportions_per_type={
            "docs_request": {"neutral": 1.0},
            "l1": {"neutral": 1.0},
            "l2": {"neutral": 1.0},
            "l3": {"neutral": 1.0},
        },
        problems=problems,
        assignment_strategy="complexity_weighted",
    )
    medium_ids = {p.id for p in problems if p.complexity == ProblemComplexity.MEDIUM}
    complex_ids = {p.id for p in problems if p.complexity == ProblemComplexity.COMPLEX}
    l2_ids = [s.problem_id for s in plan.slots]
    n_medium = sum(1 for pid in l2_ids if pid in medium_ids)
    n_complex = sum(1 for pid in l2_ids if pid in complex_ids)
    assert n_medium == 6
    assert n_complex == 4


def test_complexity_weighted_renormalizes_when_one_pref_empty() -> None:
    # L2 prefers (medium, complex); only complex present -> all slots from complex.
    problems = _problems({ProblemComplexity.COMPLEX: 4})
    plan = build_allocation_plan(
        total=10,
        type_proportions={"docs_request": 0.0, "l1": 0.0, "l2": 1.0, "l3": 0.0},
        tier_proportions={"standard": 1.0},
        tone_proportions_per_type={
            "docs_request": {"neutral": 1.0},
            "l1": {"neutral": 1.0},
            "l2": {"neutral": 1.0},
            "l3": {"neutral": 1.0},
        },
        problems=problems,
        assignment_strategy="complexity_weighted",
    )
    complex_ids = {p.id for p in problems}
    assert all(s.problem_id in complex_ids for s in plan.slots)


def test_complexity_weighted_falls_back_when_all_pref_empty() -> None:
    # L2 prefers (medium, complex); only simple present -> global fallback to simple.
    problems = _problems({ProblemComplexity.SIMPLE: 4})
    plan = build_allocation_plan(
        total=10,
        type_proportions={"docs_request": 0.0, "l1": 0.0, "l2": 1.0, "l3": 0.0},
        tier_proportions={"standard": 1.0},
        tone_proportions_per_type={
            "docs_request": {"neutral": 1.0},
            "l1": {"neutral": 1.0},
            "l2": {"neutral": 1.0},
            "l3": {"neutral": 1.0},
        },
        problems=problems,
        assignment_strategy="complexity_weighted",
    )
    simple_ids = {p.id for p in problems}
    assert all(s.problem_id in simple_ids for s in plan.slots)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/test_allocator.py::test_complexity_weighted_uses_both_preferred_buckets -v`
Expected: FAIL — current code puts all 10 L2 slots in the first non-empty bucket (medium), so `n_complex == 0`, asserting `n_medium == 6` fails (it is 10).

- [ ] **Step 3: Rewrite `_assign_complexity_weighted`**

In `src/csfd/allocator.py`, replace the entire `_assign_complexity_weighted` function (currently ~lines 174-217) with the version below. Add the module-level `_PREF_WEIGHTS` constant directly above the function. `largest_remainder` is already imported at the top of the file (`from csfd.utils.rng import derive_rng, largest_remainder`).

```python
# Rank-decay weights for splitting a ticket type's slots across the
# non-empty complexity buckets in its preference tuple. Keyed by the number
# of non-empty preferred buckets. First preferred bucket gets the largest share.
_PREF_WEIGHTS: dict[int, tuple[float, ...]] = {
    1: (1.0,),
    2: (0.6, 0.4),
    3: (0.5, 0.3, 0.2),
}


def _assign_complexity_weighted(
    *,
    type_counts: Mapping[TicketType, int],
    problems: Sequence[ProblemRef],
) -> dict[TicketType, list[str]]:
    """Assign problems by spreading each type's slots across ALL non-empty
    preferred complexity buckets, weighted by preference rank.

    For each ticket type, take the non-empty buckets named in its preference
    tuple (in order), split the type's slot count across them via rank-decay
    weights and largest-remainder rounding, and round-robin problem ids within
    each bucket. If no preferred bucket is populated, fall back to the first
    non-empty bucket in global order (simple -> medium -> complex).

    Deterministic with no seed: identical config produces identical assignments.
    """
    buckets: dict[ProblemComplexity, list[str]] = {
        ProblemComplexity.SIMPLE: [],
        ProblemComplexity.MEDIUM: [],
        ProblemComplexity.COMPLEX: [],
    }
    for p in sorted(problems, key=lambda x: x.id):
        buckets[p.complexity].append(p.id)

    fallback_order = (
        ProblemComplexity.SIMPLE,
        ProblemComplexity.MEDIUM,
        ProblemComplexity.COMPLEX,
    )

    def _round_robin(ids: list[str], count: int) -> list[str]:
        return [ids[i % len(ids)] for i in range(count)] if ids else []

    out: dict[TicketType, list[str]] = {}
    for tt, n in type_counts.items():
        pref = TICKET_TYPE_METADATA[tt].complexity_preference

        # Non-empty preferred buckets, in preference order, de-duplicated.
        non_empty_pref: list[ProblemComplexity] = []
        seen: set[ProblemComplexity] = set()
        for c in pref:
            if c in seen:
                continue
            seen.add(c)
            if buckets[c]:
                non_empty_pref.append(c)

        if non_empty_pref:
            weights = _PREF_WEIGHTS[len(non_empty_pref)]
            weight_map = {
                c.value: w for c, w in zip(non_empty_pref, weights, strict=True)
            }
            split = largest_remainder(weight_map, n)
            ordered: list[str] = []
            for c in non_empty_pref:
                ordered.extend(_round_robin(buckets[c], split[c.value]))
            out[tt] = ordered
            continue

        # All preferred buckets empty: fall back to first non-empty global bucket.
        fallback_ids: list[str] = []
        for c in fallback_order:
            if buckets[c]:
                fallback_ids = buckets[c]
                break
        out[tt] = _round_robin(fallback_ids, n)
    return out
```

- [ ] **Step 4: Update the caller to drop the now-unused `seed` argument**

`_assign_complexity_weighted` no longer takes `seed`. In `_assign_problems` (~line 149-150), update the `complexity_weighted` branch. Replace:

```python
    if strategy == "complexity_weighted":
        return _assign_complexity_weighted(type_counts=type_counts, problems=problems)
```

It already passes only `type_counts` and `problems` — confirm no `seed=` is passed in that branch. (The `uniform` branch keeps its `seed=seed`.) No change needed if it already matches; otherwise remove any `seed=` kwarg from the `complexity_weighted` call.

- [ ] **Step 5: Run the full allocator test file**

Run: `uv run pytest tests/unit/test_allocator.py -v`
Expected: PASS — the three new tests pass, and all pre-existing tests (including `test_complexity_weighted_prefers_complex_for_l3`, `test_complexity_weighted_prefers_simple_for_docs`, `test_plan_deterministic_across_calls`) still pass. (The two `prefers_*` tests exercise single-non-empty-preferred-bucket cases, which the new code handles identically.)

- [ ] **Step 6: Type-check and lint**

Run: `uv run mypy src tests && uv run ruff check src tests`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/csfd/allocator.py tests/unit/test_allocator.py
git commit -m "feat(allocator): spread complexity_weighted slots across all preferred buckets"
```

---

### Task 2: Deeper Phase 1 dedup embedding template

**Files:**
- Modify: `src/csfd/embeddings/text.py`
- Modify: `src/csfd/settings.py:76`
- Modify: `config/profiles/local-only.yaml`, `config/profiles/mixed.yaml`
- Test: `tests/unit/test_embeddings/test_text.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_embeddings/test_text.py` (the existing `_fixture()` helper builds a `ProblemBrainstormOutput` with empty `symptoms`/`root_cause`):

```python
def _fixture_with_lists() -> ProblemBrainstormOutput:
    return ProblemBrainstormOutput(
        title="Compressor surge alarm",
        summary="CT-500 trips surge alarm on cold start.",
        background="Detailed background paragraph.",
        symptoms=["alarm at startup", "compressor cycles"],
        root_cause=["faulty pressure sensor"],
        category="hardware",
        complexity=ProblemComplexity.MEDIUM,
        resolution_hints={"docs_request": "x", "l1": "x", "l2": "x", "l3": "x"},
    )


def test_symptoms_root_cause_template_includes_lists() -> None:
    text = text_for_problem(_fixture_with_lists(), "title_summary_symptoms_root_cause")
    assert text == (
        "Compressor surge alarm\n\n"
        "CT-500 trips surge alarm on cold start.\n\n"
        "Symptoms:\n- alarm at startup\n- compressor cycles\n\n"
        "Root cause:\n- faulty pressure sensor"
    )


def test_symptoms_root_cause_template_omits_empty_sections() -> None:
    # _fixture() has empty symptoms/root_cause -> degrades to title+summary only.
    text = text_for_problem(_fixture(), "title_summary_symptoms_root_cause")
    assert text == "Compressor surge alarm\n\nCT-500 trips surge alarm on cold start."
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/test_embeddings/test_text.py::test_symptoms_root_cause_template_includes_lists -v`
Expected: FAIL with `ValueError: unknown template: 'title_summary_symptoms_root_cause'`.

- [ ] **Step 3: Add the template and extend the protocol in `text.py`**

Replace the full contents of `src/csfd/embeddings/text.py` with:

```python
"""Single source of truth for the text fed into the embedding model.

Both Phase 1 generation and any downstream re-embedding must call this so the
stored vectors and the live candidate vectors derive from identical strings.
"""

from __future__ import annotations

from typing import Literal, Protocol

TextTemplate = Literal[
    "title_summary",
    "title_summary_background",
    "title_summary_symptoms_root_cause",
]


class _ProblemLike(Protocol):
    title: str
    summary: str
    background: str
    symptoms: list[str]
    root_cause: list[str]


def text_for_problem(p: _ProblemLike, template: TextTemplate) -> str:
    if template == "title_summary":
        return f"{p.title}\n\n{p.summary}"
    if template == "title_summary_background":
        return f"{p.title}\n\n{p.summary}\n\n{p.background}"
    if template == "title_summary_symptoms_root_cause":
        parts = [f"{p.title}\n\n{p.summary}"]
        if p.symptoms:
            parts.append("Symptoms:\n" + "\n".join(f"- {s}" for s in p.symptoms))
        if p.root_cause:
            parts.append("Root cause:\n" + "\n".join(f"- {c}" for c in p.root_cause))
        return "\n\n".join(parts)
    raise ValueError(f"unknown template: {template!r}")
```

- [ ] **Step 4: Mirror the Literal in `settings.py`**

In `src/csfd/settings.py`, change line 76 from:

```python
    text_template: Literal["title_summary", "title_summary_background"] = "title_summary"
```

to:

```python
    text_template: Literal[
        "title_summary",
        "title_summary_background",
        "title_summary_symptoms_root_cause",
    ] = "title_summary"
```

- [ ] **Step 5: Run the text tests**

Run: `uv run pytest tests/unit/test_embeddings/test_text.py -v`
Expected: PASS — both new tests pass and the three existing template tests still pass.

- [ ] **Step 6: Default the new template in the embedding profiles**

In `config/profiles/local-only.yaml`, change the `embedding` block from:

```yaml
embedding:
  enabled: true
```

to:

```yaml
embedding:
  enabled: true
  text_template: title_summary_symptoms_root_cause
```

In `config/profiles/mixed.yaml`, change the `embedding` block from:

```yaml
embedding:
  enabled: true
```

to:

```yaml
embedding:
  enabled: true
  text_template: title_summary_symptoms_root_cause
```

Leave `config/default.yaml` unchanged (stays on `title_summary`).

- [ ] **Step 7: Type-check, lint, and run the embeddings test dir**

Run: `uv run mypy src tests && uv run ruff check src tests && uv run pytest tests/unit/test_embeddings -q`
Expected: no type/lint errors; all embedding tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/csfd/embeddings/text.py src/csfd/settings.py config/profiles/local-only.yaml config/profiles/mixed.yaml tests/unit/test_embeddings/test_text.py
git commit -m "feat(embeddings): add symptoms+root_cause dedup template, default it in embedding profiles"
```

---

### Task 3: Reproducibility contract in README

**Files:**
- Modify: `README.md` (`## Reproducibility` section ~lines 200-214; `## Planned improvements` ~lines 281-289; `complexity_weighted` mention ~line 203)

No tests — prose only.

- [ ] **Step 1: Rewrite the Reproducibility section opening**

In `README.md`, the `## Reproducibility` section currently opens with "Every run stamps:" followed by a bullet list, then a "Determinism guarantee:" paragraph. Insert a new contract paragraph immediately under the `## Reproducibility` heading, **before** "Every run stamps:":

```markdown
**Reproducibility contract.** Given identical config, `run_seed`, and a fresh database, `csfd` reproduces the **allocation plan and slot-by-slot lineage** plus all provenance stamps — but **not generated text**. Agent temperatures are nonzero and provider models drift, so problem and resolution *content* is never bitwise-reproducible across runs. What is reproducible is the deterministic skeleton: which problem index, ticket type, tier, and tone occupy each slot, and the recorded config/prompt/model/git provenance that lets you trace any example back to its inputs.
```

- [ ] **Step 2: Update the `complexity_weighted` bullet for the new allocator behavior**

In the same section, the bullet currently reads:

```markdown
- `run_seed` — drives the `uniform` allocator's tiebreaker shuffle; the `complexity_weighted` strategy is reproducible without a seed
```

Replace it with:

```markdown
- `run_seed` — drives the `uniform` allocator's tiebreaker shuffle; the `complexity_weighted` strategy splits each ticket type's slots across its non-empty preferred complexity buckets via fixed rank weights, so it is reproducible without a seed
```

- [ ] **Step 3: Document the model-revision provenance gap**

The section ends with the embedding-scores paragraph (~line 214). Add a new paragraph immediately after it:

```markdown
Known limitation — model revisions: `agent_traces.model_id` records the model *name* (e.g. `claude-sonnet-4-6`), not the provider's underlying snapshot/revision. A provider-side model update served under the same name changes outputs without changing the recorded provenance. For stricter provenance, pin an explicit model snapshot identifier in your profile config.
```

- [ ] **Step 4: Remove the stale "Planned improvements" item**

In `## Planned improvements`, delete item **3 ("Turn-based ticket creation.")** in full — that design is already implemented as the current Phase 2 dialogue (see `docs/superpowers/specs/2026-05-27-turn-based-dialogue-design.md`). Keep items 1 (prompt optimization) and 2 (noise agent). Renumber so the list reads 1, 2.

- [ ] **Step 5: Verify the edits read cleanly**

Run: `uv run python -c "import pathlib; t = pathlib.Path('README.md').read_text(); assert 'Reproducibility contract' in t; assert 'Turn-based ticket creation' not in t; assert 'Known limitation — model revisions' in t; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: clarify reproducibility contract and drop shipped planned-improvement"
```

---

## Notes for the implementer

- All commits run pre-commit hooks (ruff, ruff-format, mypy, yaml/whitespace checks). If a hook reformats a file, re-stage and create a **new** commit — do not `--amend`.
- Tasks are independent; they can be implemented and committed in any order, but the order above (allocator → dedup → docs) groups the code changes first.
- Do not add config knobs for the rank weights — they are intentionally hardcoded (`_PREF_WEIGHTS`).
