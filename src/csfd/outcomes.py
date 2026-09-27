"""Honest case outcomes: the state the problem is left in, and how each contact ended.

``resolved`` used to mean "the last speaker said resolved"; a promised technician
visit counted as resolved. The two things are now kept apart:

* :class:`ProblemState` — the state of the customer's problem after a contact (for
  the case: after its last contact). It is planned per case before any dialogue
  (see :func:`plan_problem_states`), and the consistency check verifies that the
  dialogue reached it.
* :class:`ContactEnding` — how one contact ended as a conversation.
* :class:`Commitment` — a promise made during a contact (who does what, by when).

``resolved`` is kept as a derived column: true only for the ``fixed_*`` states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel

from csfd.utils.rng import derive_rng, largest_remainder, weighted_choice


class ProblemState(StrEnum):
    FIXED_VERIFIED = "fixed_verified"  # fixed, and the customer confirmed it works
    FIXED_UNVERIFIED = "fixed_unverified"  # fix applied, confirmation still to come
    WORKAROUND = "workaround"  # the customer can work, the cause is not removed
    PENDING_VISIT = "pending_visit"  # a technician visit is booked
    PENDING_PART = "pending_part"  # a spare or swap unit is on its way
    PENDING_CUSTOMER_TEST = "pending_customer_test"  # the customer runs a check and reports back
    ESCALATED_OPEN = "escalated_open"  # handed to a higher tier, still open
    NOT_A_FAULT = "not_a_fault"  # the machine works as designed (handling, expectation)
    ABANDONED = "abandoned"  # the customer gave up before the problem was solved


FIXED_STATES = frozenset({ProblemState.FIXED_VERIFIED, ProblemState.FIXED_UNVERIFIED})


class ContactEnding(StrEnum):
    AGREED_NEXT_STEP = "agreed_next_step"
    CUSTOMER_SATISFIED = "customer_satisfied"
    CUSTOMER_FRUSTRATED = "customer_frustrated"
    DROPPED = "dropped"
    CAP_HIT = "cap_hit"


class Commitment(BaseModel):
    """A promise made in a contact, recorded by the speaker who made it."""

    who: str  # "agent", "customer", or a named third party (e.g. "field technician")
    what: str
    due: str | None = None  # as said in the dialogue ("tomorrow 08:00", "within 24 hours")


def is_resolved(state: ProblemState | str | None) -> bool:
    """The derived ``resolved`` flag: the problem is fixed (verified or not)."""
    return state is not None and ProblemState(state) in FIXED_STATES


def planned_ending(state: ProblemState | None, end_mode: str) -> ContactEnding:
    """How a contact is planned to end, from its round end mode and planned state."""
    if end_mode == "dropped":
        return ContactEnding.DROPPED
    if end_mode == "follow_up" or state is None:
        return ContactEnding.AGREED_NEXT_STEP
    if state in FIXED_STATES or state in (ProblemState.WORKAROUND, ProblemState.NOT_A_FAULT):
        return ContactEnding.CUSTOMER_SATISFIED
    if state == ProblemState.ABANDONED:
        return ContactEnding.CUSTOMER_FRUSTRATED
    return ContactEnding.AGREED_NEXT_STEP


# How a speaker's ``done_reason`` maps onto a contact ending.
_DONE_REASON_ENDINGS: dict[str, ContactEnding] = {
    "follow_up": ContactEnding.AGREED_NEXT_STEP,
    "escalation": ContactEnding.AGREED_NEXT_STEP,
    "resolved": ContactEnding.CUSTOMER_SATISFIED,
    "customer_satisfied": ContactEnding.CUSTOMER_SATISFIED,
    "issue_fixed": ContactEnding.CUSTOMER_SATISFIED,
    "customer_frustrated": ContactEnding.CUSTOMER_FRUSTRATED,
}


def observed_ending(end_reason: str, last_done_reason: str | None) -> ContactEnding:
    """How a contact actually ended, from the loop's end reason and the last ``done_reason``."""
    if end_reason == "dropped":
        return ContactEnding.DROPPED
    if end_reason == "cap_hit":
        return ContactEnding.CAP_HIT
    return _DONE_REASON_ENDINGS.get(last_done_reason or "", ContactEnding.AGREED_NEXT_STEP)


def plan_problem_states(
    slot_indices: Sequence[int],
    viable: Mapping[int, Sequence[str]],
    proportions: Mapping[str, float],
    *,
    seed: int,
) -> dict[int, ProblemState]:
    """Plan each case's final problem state from ``proportions``, within its problem's options.

    ``viable`` maps a slot to the states its problem allows (Phase 1's
    ``viable_outcomes``; empty means any). The proportions become exact counts
    (largest remainder); slots, in a seeded order, then take the viable state
    with the most quota left. A slot none of whose viable states has quota
    left falls back to a seeded weighted pick among them (or its first viable
    state when none is weighted), so counts are exact whenever viability allows.
    """
    weighted = {k: w for k, w in proportions.items() if w > 0}
    quota = largest_remainder(weighted, len(slot_indices)) if weighted else {}
    order = list(slot_indices)
    derive_rng(seed, "cases:state-order").shuffle(order)
    out: dict[int, ProblemState] = {}
    for index in order:
        allowed = [ProblemState(s) for s in viable.get(index, ())] or list(ProblemState)
        open_states = [s for s in allowed if quota.get(s.value, 0) > 0]
        if open_states:
            state = max(open_states, key=lambda s: quota[s.value])
            quota[state.value] -= 1
        else:
            weights: dict[str, float] = {
                s.value: weighted[s.value] for s in allowed if s.value in weighted
            }
            rng = derive_rng(seed, f"case:{index}:state")
            state = ProblemState(weighted_choice(weights, rng)) if weights else allowed[0]
        out[index] = state
    return out
