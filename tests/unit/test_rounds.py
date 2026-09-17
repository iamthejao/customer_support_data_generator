"""Unit tests for the deterministic multi-contact round plan (``csfd.rounds``)."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from csfd.calls import describe_gap, schedule_next_contact
from csfd.rounds import RoundSpec, assign_round_counts, contact_label, plan_case_rounds
from csfd.settings import CalendarConfig, CallbackReason, RoundsConfig
from csfd.utils.rng import derive_rng


def test_assign_round_counts_matches_proportions_exactly_and_is_seeded() -> None:
    slots = list(range(1, 11))
    counts = assign_round_counts(slots, {1: 0.6, 2: 0.3, 3: 0.1}, seed=42)
    assert set(counts) == set(slots)
    assert Counter(counts.values()) == {1: 6, 2: 3, 3: 1}
    assert assign_round_counts(slots, {1: 0.6, 2: 0.3, 3: 0.1}, seed=42) == counts
    others = [assign_round_counts(slots, {1: 0.6, 2: 0.3, 3: 0.1}, seed=s) for s in range(5)]
    assert any(o != counts for o in others)


def test_assign_round_counts_default_is_single_contact() -> None:
    assert assign_round_counts([1, 2, 3], RoundsConfig().proportions, seed=0) == {1: 1, 2: 1, 3: 1}


def test_contact_label_keeps_single_contact_label_for_round_one() -> None:
    assert contact_label(7, 1) == "contact:7"
    assert contact_label(7, 3) == "contact:7:r3"


def _plan(
    count: int,
    *,
    seed: int = 1,
    reasons: dict[CallbackReason, float] | None = None,
    cap: int = 20,
) -> list[RoundSpec]:
    rounds = RoundsConfig(
        proportions={count: 1.0},
        callback_reasons=reasons or {"follow_up": 0.5, "dropped": 0.5},
    )
    return plan_case_rounds(
        slot_index=3,
        round_count=count,
        agent_name="Agent-l2-0003",
        rounds=rounds,
        calendar=CalendarConfig(),
        seed=seed,
        opening_turns=2,
        turn_cap=cap,
    )


def test_plan_case_rounds_orders_contacts_and_ends_with_final() -> None:
    specs = _plan(4)
    assert [s.sequence for s in specs] == [1, 2, 3, 4]
    assert all(s.count == 4 for s in specs)
    assert specs[-1].end_mode == "final"
    assert specs[-1].drop_after_turns is None
    assert all(s.end_mode in ("follow_up", "dropped") for s in specs[:-1])
    assert [s.agent_name for s in specs] == [
        "Agent-l2-0003",
        "Agent-l2-0003-r2",
        "Agent-l2-0003-r3",
        "Agent-l2-0003-r4",
    ]
    for prev, nxt in pairwise(specs):
        assert nxt.started_at - prev.started_at >= timedelta(hours=2)
        assert nxt.started_at.weekday() < 5
    assert _plan(4) == specs


def test_plan_case_rounds_single_contact_is_final() -> None:
    (only,) = _plan(1)
    assert only.end_mode == "final"
    assert only.agent_name == "Agent-l2-0003"


@pytest.mark.parametrize("seed", range(20))
def test_dropped_contacts_keep_an_agent_reply_and_stay_under_the_cap(seed: int) -> None:
    for spec in _plan(3, seed=seed, reasons={"dropped": 1.0}, cap=6)[:-1]:
        assert spec.end_mode == "dropped"
        assert spec.drop_after_turns is not None
        assert 3 <= spec.drop_after_turns <= 5


def test_follow_up_only_reasons_never_drop() -> None:
    specs = _plan(5, reasons={"follow_up": 1.0, "dropped": 0.0})
    assert {s.end_mode for s in specs[:-1]} == {"follow_up"}


def test_schedule_next_contact_rolls_into_business_hours() -> None:
    cal = CalendarConfig()
    friday_evening = datetime(2026, 1, 9, 17, 0, tzinfo=UTC)
    for i in range(30):
        nxt = schedule_next_contact(
            friday_evening, gap_hours=(1.0, 6.0), calendar=cal, rng=derive_rng(i, "g")
        )
        # Any gap from a Friday 17:00 start lands after hours -> Monday morning.
        assert nxt.date() == datetime(2026, 1, 12).date()
        assert 8 <= nxt.hour < 10


def test_schedule_next_contact_keeps_in_hours_gap() -> None:
    start = datetime(2026, 1, 6, 9, 0, tzinfo=UTC)
    nxt = schedule_next_contact(
        start, gap_hours=(2.0, 2.0), calendar=CalendarConfig(), rng=derive_rng(0, "g")
    )
    assert nxt == start + timedelta(hours=2)


@pytest.mark.parametrize(
    ("seconds", "phrase"),
    [(600, "about 10 minutes"), (3 * 3600, "about 3 hours"), (50 * 3600, "about 2 days")],
)
def test_describe_gap(seconds: float, phrase: str) -> None:
    assert describe_gap(seconds) == phrase


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"proportions": {0: 1.0}}, "must be >= 1"),
        ({"proportions": {1: 0.0}}, "positive sum"),
        ({"callback_reasons": {"dropped": 0.0}}, "positive sum"),
        ({"gap_hours": (0.5, 4.0)}, "gap_hours"),
        ({"gap_hours": (5.0, 4.0)}, "gap_hours"),
    ],
)
def test_rounds_config_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RoundsConfig.model_validate(kwargs)
