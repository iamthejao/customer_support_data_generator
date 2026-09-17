"""Deterministic round plan: how one case spreads over several contacts.

A *case* is one allocation slot (see :mod:`csfd.allocator`). With
``tickets.rounds`` configured, a case yields N related contacts ("rounds") —
e.g. a caller who hangs up or is asked to try something, then calls back.
This module decides, before any LLM call and purely from
``(run_seed, slot_index)`` and config:

* how many rounds each case gets (largest-remainder counts, seeded shuffle);
* when each round starts (see :func:`csfd.calls.schedule_next_contact`);
* how each non-final round ends (``follow_up`` or ``dropped``) and, for a
  dropped call, after how many turns the line cuts off;
* which agent picks up each round.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from csfd.calls import schedule_first_contact, schedule_next_contact
from csfd.settings import CalendarConfig, RoundsConfig
from csfd.utils.rng import derive_rng, largest_remainder, weighted_choice

EndMode = Literal["final", "follow_up", "dropped"]

# A dropped call is cut off after the opening plus this many further turns.
_DROP_AFTER_EXTRA_TURNS = (1, 4)


@dataclass(slots=True, frozen=True)
class RoundSpec:
    """One planned contact of a case."""

    sequence: int  # 1-based position within the case
    count: int  # total contacts in the case
    started_at: datetime
    agent_name: str
    end_mode: EndMode
    drop_after_turns: int | None = None


def contact_label(slot_index: int, sequence: int) -> str:
    """Stable per-contact label for RNG streams; round 1 keeps the single-contact label."""
    return f"contact:{slot_index}" if sequence == 1 else f"contact:{slot_index}:r{sequence}"


def assign_round_counts(
    slot_indices: Sequence[int], proportions: Mapping[int, float], *, seed: int
) -> dict[int, int]:
    """Map each slot index to its number of rounds, matching ``proportions`` exactly."""
    ordered = sorted(proportions)
    counts = largest_remainder({str(k): proportions[k] for k in ordered}, len(slot_indices))
    values = [k for k in ordered for _ in range(counts[str(k)])]
    derive_rng(seed, "rounds:assign").shuffle(values)
    return dict(zip(slot_indices, values, strict=True))


def plan_case_rounds(
    *,
    slot_index: int,
    round_count: int,
    agent_name: str,
    rounds: RoundsConfig,
    calendar: CalendarConfig,
    seed: int,
    opening_turns: int,
    turn_cap: int,
) -> list[RoundSpec]:
    """Plan every contact of one case.

    ``opening_turns`` is how many turns the contact opens with before the
    agent's first reply (1 for email, 2 for phone: greeting + caller). A
    dropped contact always keeps at least one agent reply and is cut before
    ``turn_cap`` would fire.
    """
    specs: list[RoundSpec] = []
    started_at: datetime | None = None
    for sequence in range(1, round_count + 1):
        label = contact_label(slot_index, sequence)
        if started_at is None:
            started_at = schedule_first_contact(calendar, derive_rng(seed, f"{label}:schedule"))
        else:
            started_at = schedule_next_contact(
                started_at,
                gap_hours=rounds.gap_hours,
                calendar=calendar,
                rng=derive_rng(seed, f"{label}:schedule"),
            )
        end_mode: EndMode = "final"
        drop_after: int | None = None
        if sequence < round_count:
            rng = derive_rng(seed, f"{label}:end")
            reasons: dict[str, float] = {k: w for k, w in rounds.callback_reasons.items() if w > 0}
            end_mode = "dropped" if weighted_choice(reasons, rng) == "dropped" else "follow_up"
            if end_mode == "dropped":
                extra = rng.randint(*_DROP_AFTER_EXTRA_TURNS)
                drop_after = max(opening_turns + 1, min(opening_turns + extra, turn_cap - 1))
        specs.append(
            RoundSpec(
                sequence=sequence,
                count=round_count,
                started_at=started_at,
                agent_name=agent_name if sequence == 1 else f"{agent_name}-r{sequence}",
                end_mode=end_mode,
                drop_after_turns=drop_after,
            )
        )
    return specs
