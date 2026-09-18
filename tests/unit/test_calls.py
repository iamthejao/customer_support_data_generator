"""Unit tests for the deterministic phone-call helpers in ``csfd.calls``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from csfd.calls import (
    estimate_email_times,
    estimate_turn_timings,
    format_offset,
    schedule_first_contact,
    scripted_greeting,
    spoken_word_count,
)
from csfd.settings import CalendarConfig
from csfd.utils.rng import derive_rng


def test_schedule_first_contact_is_deterministic_weekday_and_in_hours() -> None:
    cal = CalendarConfig(start=datetime(2026, 1, 3, 8, 0, tzinfo=UTC), span_days=15)  # Saturday
    starts = [
        schedule_first_contact(cal, derive_rng(42, f"contact:{i}:schedule")) for i in range(50)
    ]
    again = [
        schedule_first_contact(cal, derive_rng(42, f"contact:{i}:schedule")) for i in range(50)
    ]
    assert starts == again
    for s in starts:
        assert s.weekday() < 5
        assert 8 <= s.hour < 18
        assert s.tzinfo is UTC
        # 15 business days from the first Monday (Jan 5) end before Jan 26.
        assert datetime(2026, 1, 5, tzinfo=UTC) <= s < datetime(2026, 1, 26, tzinfo=UTC)
    assert len(set(starts)) > 40


def test_calendar_rejects_inverted_business_hours() -> None:
    with pytest.raises(ValueError, match="business_hours"):
        CalendarConfig(business_hours=(18, 8))


def test_scripted_greeting_names_company_and_agent() -> None:
    at = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)
    greetings = {
        scripted_greeting(company="CoolTherm", agent="Agent-l1-0001", at=at, rng=derive_rng(1, s))
        for s in map(str, range(40))
    }
    assert len(greetings) > 1
    for g in greetings:
        assert "CoolTherm" in g
        assert "Agent-l1-0001" in g
        assert "{" not in g
        assert "afternoon" not in g
    evening = scripted_greeting(
        company="C", agent="A", at=at + timedelta(hours=9), rng=derive_rng(0, "x")
    )
    assert "morning" not in evening


def test_spoken_word_count_ignores_tags_and_cutoffs() -> None:
    assert spoken_word_count("[hold] Thanks for holding --") == 3
    assert spoken_word_count("um [pause] okay [inaudible] yes") == 3


def test_turn_timings_are_ordered_and_reflect_speech_length() -> None:
    contents = ["Hello, support.", "Hi " + "word " * 25, "Okay."]
    timings = estimate_turn_timings(contents, derive_rng(7, "t"))
    assert timings[0].start_s == 0.0
    assert [t.start_s for t in timings] == sorted(t.start_s for t in timings)
    # 26 words at 2.5 words/s is ~10.4s of speech.
    assert timings[1].end_s - timings[1].start_s == pytest.approx(10.4, abs=0.2)
    assert estimate_turn_timings(contents, derive_rng(7, "t")) == timings


def test_turn_timings_hold_pause_and_interruption() -> None:
    contents = [
        "Can you hold a moment?",
        "Sure.",
        "[hold] Thanks for holding, so the --",
        "Sorry, the unit just [pause] restarted.",
    ]
    t = estimate_turn_timings(contents, derive_rng(3, "t"))
    assert t[2].hold_s >= 30.0
    assert t[2].start_s >= t[1].end_s + 30.0
    # The caller cut in: their turn starts before the agent finished.
    assert t[3].start_s < t[2].end_s
    assert t[3].start_s > t[2].start_s
    # [pause] adds at least 1.5s on top of ~2.4s of speech (6 words).
    assert t[3].end_s - t[3].start_s >= 3.9


def test_format_offset() -> None:
    assert format_offset(0) == "00:00:00"
    assert format_offset(3725.9) == "01:02:05"


def test_email_times_are_ordered_in_hours_and_deterministic() -> None:
    cal = CalendarConfig()
    start = datetime(2026, 1, 9, 16, 50, tzinfo=UTC)  # Friday late afternoon
    speakers = ["customer", "agent", "customer", "agent", "customer", "agent"]
    times = estimate_email_times(speakers, started_at=start, calendar=cal, rng=derive_rng(5, "e"))
    assert times[0] == start
    assert times == sorted(times)
    assert len(times) == len(speakers)
    for t in times[1:]:
        assert t.weekday() < 5
        assert 8 <= t.hour < 18
    assert times == estimate_email_times(
        speakers, started_at=start, calendar=cal, rng=derive_rng(5, "e")
    )


def test_email_thread_is_compressed_before_the_next_contact() -> None:
    start = datetime(2026, 1, 6, 9, 0, tzinfo=UTC)
    nxt = start + timedelta(hours=3)
    speakers = ["customer", "agent"] * 6
    for seed in range(20):
        times = estimate_email_times(
            speakers,
            started_at=start,
            calendar=CalendarConfig(),
            rng=derive_rng(seed, "e"),
            next_contact_at=nxt,
        )
        assert times[0] == start
        assert times == sorted(times)
        assert times[-1] <= start + timedelta(hours=3) * 0.8


def test_late_afternoon_thread_stays_in_hours_when_compressed() -> None:
    """A thread opened minutes before closing still dates every message in hours."""
    cal = CalendarConfig()
    start = datetime(2026, 1, 5, 17, 29, tzinfo=UTC)  # Monday, one minute of the day left
    nxt = datetime(2026, 1, 6, 8, 30, tzinfo=UTC)  # callback the next business morning
    speakers = ["customer", "agent"] * 4
    for seed in range(20):
        times = estimate_email_times(
            speakers,
            started_at=start,
            calendar=cal,
            rng=derive_rng(seed, "e"),
            next_contact_at=nxt,
        )
        assert times[0] == start
        assert times == sorted(times)
        assert times[-1] < nxt
        for t in times:
            assert t.weekday() < 5, t
            assert 8 <= t.hour < 18, t
