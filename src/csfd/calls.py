"""Deterministic contact metadata for phone calls and email threads.

The LLM agents write only what each speaker says. Everything a telephony or
mail system would record around the words — when the contact started, the
agent's scripted greeting, when each utterance starts and ends, when each email
was sent — is derived here from the text plus a seeded RNG (see
:func:`csfd.utils.rng.derive_rng`), never from the wall clock and never asked
of the model:

* :func:`schedule_first_contact` / :func:`schedule_next_contact` — weekday,
  business-hours start times for a case's first and follow-up contacts.
* :func:`scripted_greeting` — the agent's opening line, picked from a small
  set of call-center scripts.
* :func:`estimate_turn_timings` — per-utterance offsets from word counts, a
  speaking rate, response latencies, and the transcript tags below.
* :func:`estimate_email_times` — a sent time per message of an email thread.

Transcript tags the phone prompts may emit (and nothing else in brackets):

* ``[hold]`` — at the start of an agent turn: the caller was on hold and the
  agent is back. Adds a hold gap before the turn.
* ``[pause]`` — a noticeable silence inside a turn.
* ``[inaudible]`` — a word or phrase lost on the line.
* ``--`` at the end of a turn — the speaker was cut off; the next speaker
  starts slightly before they finished.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random

from csfd.settings import CalendarConfig

# Conversational English runs at roughly 150 words per minute.
WORDS_PER_SECOND = 2.5
_RESPONSE_LATENCY_S = (0.3, 1.5)
_INTERRUPT_OVERLAP_S = (0.2, 0.8)
_HOLD_S = (30.0, 180.0)
_PAUSE_S = (1.5, 4.0)
# Minutes before the next email in a thread (log-uniform): support answers
# within hours, customers take a little longer.
_EMAIL_REPLY_MIN = {"agent": (5.0, 240.0), "customer": (3.0, 480.0)}
# An email thread planned before another contact ends within this share of the gap.
_EMAIL_DEADLINE_SHARE = 0.8

HOLD_TAG = "[hold]"
_TAG_RE = re.compile(r"\[(?:hold|pause|inaudible|crosstalk)\]", re.IGNORECASE)
_PAUSE_RE = re.compile(r"\[pause\]", re.IGNORECASE)

_GREETINGS: tuple[str, ...] = (
    "Thank you for calling {company} support, this is {agent}. How can I help you today?",
    "{company} customer service, {agent} speaking. What can I do for you?",
    "Good {daypart}, you've reached {company} technical support. "
    "My name is {agent}, how can I help?",
    "Hi, thanks for calling {company}. This is {agent}, how can I help you today?",
)


@dataclass(slots=True, frozen=True)
class TurnTiming:
    """Offsets in seconds from the start of the call."""

    start_s: float
    end_s: float
    hold_s: float = 0.0


def _is_business_day(d: datetime) -> bool:
    return d.weekday() < 5


def _add_business_days(start: datetime, days: int) -> datetime:
    d = start
    while not _is_business_day(d):
        d += timedelta(days=1)
    remaining = days
    while remaining > 0:
        d += timedelta(days=1)
        if _is_business_day(d):
            remaining -= 1
    return d


def schedule_first_contact(calendar: CalendarConfig, rng: Random) -> datetime:
    """Pick a weekday, in-hours start time within ``calendar.span_days`` business days."""
    open_h, close_h = calendar.business_hours
    day = _add_business_days(calendar.start, rng.randrange(calendar.span_days))
    # Leave half an hour before closing so the call itself fits in hours.
    window_min = max(1, (close_h - open_h) * 60 - 30)
    minute = rng.randrange(window_min)
    midnight = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(hours=open_h, minutes=minute, seconds=rng.randrange(60))


def _in_business_hours(at: datetime, calendar: CalendarConfig) -> bool:
    open_h, close_h = calendar.business_hours
    minutes = at.hour * 60 + at.minute
    return _is_business_day(at) and open_h * 60 <= minutes < close_h * 60 - 30


def schedule_next_contact(
    previous_start: datetime,
    *,
    gap_hours: tuple[float, float],
    calendar: CalendarConfig,
    rng: Random,
) -> datetime:
    """Start time for the next contact of a case, at least ``gap_hours[0]`` later.

    The gap is drawn log-uniformly (many same-day callbacks, a tail of
    multi-day ones). A time outside business hours moves to the next business
    morning, which only ever makes the gap longer.
    """
    lo, hi = gap_hours
    gap = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return _roll_into_business_hours(previous_start + timedelta(hours=gap), calendar, rng)


def _roll_into_business_hours(at: datetime, calendar: CalendarConfig, rng: Random) -> datetime:
    """Keep ``at`` if it is in business hours, else move it to the next business morning."""
    morning_jitter = timedelta(minutes=rng.randrange(90), seconds=rng.randrange(60))
    if _in_business_hours(at, calendar):
        return at
    open_h = calendar.business_hours[0]
    day = at if at.hour < open_h else at + timedelta(days=1)
    while not _is_business_day(day):
        day += timedelta(days=1)
    return day.replace(hour=open_h, minute=0, second=0, microsecond=0) + morning_jitter


def estimate_email_times(
    speakers: Sequence[str],
    *,
    started_at: datetime,
    calendar: CalendarConfig,
    rng: Random,
    next_contact_at: datetime | None = None,
) -> list[datetime]:
    """Sent time of each message in an email thread, starting at ``started_at``.

    Each reply follows the previous message after a log-uniform delay and is
    moved into business hours. When another contact of the same case is planned
    at ``next_contact_at``, the thread is compressed to finish well before it,
    so contacts never interleave and planned start times stay untouched.
    """
    sent = [started_at]
    for speaker in speakers[1:]:
        lo, hi = _EMAIL_REPLY_MIN.get(speaker, _EMAIL_REPLY_MIN["customer"])
        delay = timedelta(minutes=math.exp(rng.uniform(math.log(lo), math.log(hi))))
        sent.append(_roll_into_business_hours(sent[-1] + delay, calendar, rng))
    if next_contact_at is not None and len(sent) > 1:
        budget = (next_contact_at - started_at) * _EMAIL_DEADLINE_SHARE
        span = sent[-1] - started_at
        if span > budget:
            sent = [started_at + (t - started_at) * (budget / span) for t in sent]
    return [t.replace(microsecond=0) for t in sent]


def _daypart(at: datetime) -> str:
    if at.hour < 12:
        return "morning"
    if at.hour < 17:
        return "afternoon"
    return "evening"


def scripted_greeting(*, company: str, agent: str, at: datetime, rng: Random) -> str:
    """Return the agent's scripted opening line for a call starting at ``at``."""
    template = _GREETINGS[rng.randrange(len(_GREETINGS))]
    return template.format(company=company, agent=agent, daypart=_daypart(at))


def spoken_word_count(text: str) -> int:
    """Words actually spoken: transcript tags and cut-off markers are not speech."""
    return len(_TAG_RE.sub(" ", text).replace("--", " ").split())


def estimate_turn_timings(contents: Sequence[str], rng: Random) -> list[TurnTiming]:
    """Estimate per-utterance start/end offsets for a call transcript.

    Deterministic for a given ``rng`` state and text. Offsets are rounded to
    tenths of a second and are monotonically non-decreasing by start time.
    """
    out: list[TurnTiming] = []
    clock = 0.0
    prev_start = 0.0
    prev_text = ""
    for i, text in enumerate(contents):
        stripped = text.strip()
        hold_s = 0.0
        if i == 0:
            start = 0.0
        elif prev_text.endswith("--"):
            # The previous speaker was cut off: this turn overlaps its tail.
            start = max(prev_start + 0.1, clock - rng.uniform(*_INTERRUPT_OVERLAP_S))
        else:
            start = clock + rng.uniform(*_RESPONSE_LATENCY_S)
        if stripped.lower().startswith(HOLD_TAG):
            hold_s = rng.uniform(*_HOLD_S)
            start += hold_s
        speech_s = max(1.0, spoken_word_count(stripped) / WORDS_PER_SECOND)
        pause_s = sum(rng.uniform(*_PAUSE_S) for _ in _PAUSE_RE.findall(stripped))
        end = start + speech_s + pause_s
        out.append(
            TurnTiming(start_s=round(start, 1), end_s=round(end, 1), hold_s=round(hold_s, 1))
        )
        clock = max(clock, end)
        prev_start = start
        prev_text = stripped
    return out


def describe_gap(seconds: float) -> str:
    """Human phrase for the time between two contacts, as a caller would put it."""
    minutes = seconds / 60
    if minutes < 90:
        return f"about {max(1, round(minutes))} minutes"
    hours = minutes / 60
    if hours < 36:
        return f"about {round(hours)} hours"
    return f"about {round(hours / 24)} days"


def format_offset(seconds: float) -> str:
    """Format a call offset or duration as ``HH:MM:SS``."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
