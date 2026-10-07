"""Documentation coverage: which candidate causes the documents describe, and how far.

Real documentation is incomplete on purpose: some fixes live only in the support
agent's experience. Each candidate cause gets one coverage value:

* ``documented`` — the cause, its checks and its fix are in the documents;
* ``partial`` — the cause and its checks are documented, its fix is not;
* ``agent_only`` — the documents never mention it; only the agent knows it.

Causes are compared by their text (lower case, punctuation removed), and the
same cause gets the same value everywhere in the run: a shared service-manual
section documents a cause for every problem that has it, so the fix of an
``agent_only`` cause must be absent from the whole corpus. Coverage is therefore
drawn once per distinct cause text, with exact counts over those causes
(largest remainder, like planned problem states) in a seeded order.

The root cause's coverage decides where a case's answer lives
(:func:`answer_source`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Literal

from csfd.diagnosis import DiagnosisPlan
from csfd.utils.rng import derive_rng, largest_remainder

Coverage = Literal["documented", "partial", "agent_only"]
AnswerSource = Literal["documents", "partial", "agent_knowledge"]

_ANSWER_SOURCES: dict[Coverage, AnswerSource] = {
    "documented": "documents",
    "partial": "partial",
    "agent_only": "agent_knowledge",
}


def default_coverage_proportions() -> dict[Coverage, float]:
    return {"documented": 0.6, "partial": 0.2, "agent_only": 0.2}


def cause_key(cause: str) -> str:
    """Cause text compared loosely: lower case, punctuation and extra spaces removed."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", cause.lower()).split())


def answer_source(coverage: Coverage) -> AnswerSource:
    """Where a case's answer lives, given its root cause's coverage."""
    return _ANSWER_SOURCES[coverage]


def plan_coverage(
    plans: Iterable[tuple[str, DiagnosisPlan]],
    proportions: Mapping[Coverage, float],
    *,
    seed: int,
) -> dict[str, dict[str, Coverage]]:
    """Coverage per problem and candidate cause (``{problem_id: {cause: coverage}}``).

    ``plans`` are ``(problem_id, plan)`` pairs; distinct cause texts are taken in
    their order of first appearance. Their count is split exactly by
    ``proportions`` (largest remainder); a seeded shuffle then decides which
    cause gets which value, and every candidate cause takes its text's value.
    """
    weighted: dict[Coverage, float] = {k: w for k, w in proportions.items() if w > 0}
    pairs = [(pid, c.cause) for pid, plan in plans for c in plan.candidate_causes]
    keys = list(dict.fromkeys(cause_key(cause) for _, cause in pairs))
    quota = largest_remainder({str(k): w for k, w in weighted.items()}, len(keys)) if keys else {}
    values: list[Coverage] = [level for level in weighted for _ in range(quota.get(level, 0))]
    order = list(keys)
    derive_rng(seed, "documents:coverage").shuffle(order)
    by_key = dict(zip(order, values, strict=True))
    out: dict[str, dict[str, Coverage]] = {}
    for pid, cause in pairs:
        out.setdefault(pid, {})[cause] = by_key[cause_key(cause)]
    return out
