"""The canonical diagnosis plan of a problem, and how a case's contacts walk through it.

Phase 1 writes one :class:`DiagnosisPlan` per problem, before any dialogue: the
causes an agent would weigh for these symptoms, each with its own fix
(resolution, parts, verification, workaround, prevention) as a knowledge-base
troubleshooting guide lists it, and the ordered checks that tell them apart (how
to check, what the customer finds on this problem, which alternatives that
result rules out, and which check confirms the cause). Exactly one cause is
marked as the root cause. It is the ground truth a conversation has lineage to.

Phase 2 does not hand the agent the root cause. The agent gets the troubleshooting
guide (:func:`agent_guide`): every candidate cause with its fix, in a neutral
order, and the checks without their results, so it has to run the checks to know
which fix applies. The customer gets what they would find when asked to do each
check (:func:`customer_findings`). Each contact follows a planned
:class:`ContactBeat` (:func:`plan_beats`): which checks it covers, whether the
cause is reached, and the problem state it leaves the case in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from csfd.outcomes import ProblemState
from csfd.utils.rng import derive_rng


class DiagnosticCheck(BaseModel):
    """One step of the diagnosis, as a support agent would run it with the customer."""

    check: str  # short name, e.g. "Vacuum self-test"
    how_to_check: str  # what the agent asks the customer to do, read out or look at
    finding: str  # what the customer observes on THIS problem (the result if the cause is present)
    rules_out: list[str] = Field(default_factory=list)  # candidate causes this result eliminates
    confirms_cause: bool = False  # this result pins down the root cause


class CandidateCause(BaseModel):
    """One cause an agent would weigh, with the fix the guide gives for it."""

    cause: str
    is_root_cause: bool = False  # the true cause of this problem (ground truth)
    resolution_steps: list[str] = Field(default_factory=list)
    parts: list[str] = Field(default_factory=list)
    verification: str = ""  # how the fix is confirmed
    verification_finding: str = ""  # what the customer sees once the fix works
    workaround: str = ""  # a way to keep working while the cause is not removed
    preventive_action: str = ""


class DiagnosisPlan(BaseModel):
    """How the problem is diagnosed and fixed, written before any conversation."""

    candidate_causes: list[CandidateCause] = Field(default_factory=list)
    checks: list[DiagnosticCheck] = Field(default_factory=list)
    safety: list[str] = Field(default_factory=list)

    def root_cause(self) -> CandidateCause | None:
        """The candidate cause marked as the true one, if any."""
        return next((c for c in self.candidate_causes if c.is_root_cause), None)

    def confirming_index(self) -> int:
        """Index of the check that confirms the cause (the last check when none is marked)."""
        marked = [i for i, c in enumerate(self.checks) if c.confirms_cause]
        return marked[0] if marked else len(self.checks) - 1


class ContactBeat(BaseModel):
    """What one contact of a case is planned to get through."""

    checks: list[int] = Field(default_factory=list)  # plan check indices run in this contact
    next_check: int | None = None  # the check agreed as the customer's test for next time
    cause_confirmed: bool = False  # the agent reaches the root cause in this contact
    problem_state: ProblemState | None = None  # state the contact leaves the case in


# Final states where the diagnosis stops before the confirming check.
_STOPS_BEFORE_CONFIRMING = frozenset(
    {
        ProblemState.PENDING_CUSTOMER_TEST,
        ProblemState.ESCALATED_OPEN,
        ProblemState.ABANDONED,
    }
)


def plan_beats(
    plan: DiagnosisPlan | None, end_modes: Sequence[str], final_state: ProblemState
) -> list[ContactBeat]:
    """Split the plan's checks over a case's contacts, in order.

    Checks run up to the confirming one; states that stop short of it
    (``pending_customer_test``, ``escalated_open``, ``abandoned``) leave the
    confirming check undone. A dropped contact is cut off too early to report
    results, so it runs no checks; the checks are spread evenly across the
    other contacts, later ones taking the remainder, so the final contact
    always has one when there is any. A ``follow_up`` contact agrees the next
    contact's first check as the customer's test and leaves the case
    ``pending_customer_test``; a dropped contact leaves no state.
    """
    count = len(end_modes)
    if plan is None or not plan.checks:
        runnable: list[int] = []
    else:
        confirm = plan.confirming_index()
        stop = confirm if final_state in _STOPS_BEFORE_CONFIRMING else confirm + 1
        runnable = list(range(stop))
    working = [p for p, mode in enumerate(end_modes) if mode != "dropped"]
    base, extra = divmod(len(runnable), len(working)) if working else (0, 0)
    assigned: list[list[int]] = [[] for _ in end_modes]
    start = 0
    for rank, position in enumerate(working):
        size = base + (1 if rank >= len(working) - extra else 0)
        assigned[position] = runnable[start : start + size]
        start += size
    beats: list[ContactBeat] = []
    for position, end_mode in enumerate(end_modes):
        final = position == count - 1
        pending: int | None = None
        if final:
            state: ProblemState | None = final_state
            if final_state == ProblemState.PENDING_CUSTOMER_TEST and plan and plan.checks:
                pending = plan.confirming_index()
        elif end_mode == "follow_up":
            state = ProblemState.PENDING_CUSTOMER_TEST
            following = assigned[position + 1]
            pending = following[0] if following else None
        else:
            state = None
        confirmed = (
            final
            and plan is not None
            and bool(plan.checks)
            and final_state not in _STOPS_BEFORE_CONFIRMING
        )
        beats.append(
            ContactBeat(
                checks=assigned[position],
                next_check=pending,
                cause_confirmed=confirmed,
                problem_state=state,
            )
        )
    return beats


def agent_guide(
    plan: DiagnosisPlan, beat: ContactBeat, *, done_checks: Sequence[int], seed_label: str
) -> dict[str, Any]:
    """The troubleshooting guide the agent works from: no root cause, no results.

    Every candidate cause comes with its own fix, in a seeded order, so neither
    its position nor the presence of a fix marks the true cause; the agent has
    to run the checks to know which fix applies.
    """
    causes = [c.model_dump(exclude={"is_root_cause"}) for c in plan.candidate_causes]
    derive_rng(0, f"{seed_label}:causes").shuffle(causes)
    return {
        "candidate_causes": causes,
        "checks": [
            {"step": i + 1, "check": c.check, "how_to_check": c.how_to_check}
            for i, c in enumerate(plan.checks)
        ],
        "done_steps": [i + 1 for i in done_checks],
        "beat_steps": [i + 1 for i in beat.checks],
        "next_step": beat.next_check + 1 if beat.next_check is not None else None,
        "safety": plan.safety,
    }


def customer_findings(
    plan: DiagnosisPlan, beat: ContactBeat, *, done_checks: Sequence[int]
) -> dict[str, Any]:
    """What the customer finds when asked to do each check, up to this contact's beat.

    Checks already run in earlier contacts are what the customer remembers;
    the agreed test for next time is left out, since it has not been done yet.
    """
    visible = [*done_checks, *beat.checks]
    out: dict[str, Any] = {
        "findings": [
            {"how_to_check": plan.checks[i].how_to_check, "finding": plan.checks[i].finding}
            for i in visible
        ],
    }
    root = plan.root_cause()
    if beat.problem_state == ProblemState.FIXED_VERIFIED and root and root.verification_finding:
        out["after_fix"] = root.verification_finding
    return out
