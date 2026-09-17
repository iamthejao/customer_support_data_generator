"""Deterministic helpers for phone-call style conversations.

The LLM agents write only what each speaker says. Everything a telephony system
would record around the words — when the call started, the agent's scripted
greeting, and when each utterance starts and ends — is derived here from the
text plus a seeded RNG (see :func:`csfd.utils.rng.derive_rng`), never from the
wall clock and never asked of the model:

* :func:`schedule_first_contact` — a weekday, business-hours start time.
* :func:`scripted_greeting` — the agent's opening line, picked from a small
  set of call-center scripts.
* :func:`estimate_turn_timings` — per-utterance offsets from word counts, a
  speaking rate, response latencies, and the transcript tags below.

Transcript tags the phone prompts may emit (and nothing else in brackets):

* ``[hold]`` — at the start of an agent turn: the caller was on hold and the
  agent is back. Adds a hold gap before the turn.
* ``[pause]`` — a noticeable silence inside a turn.
* ``[inaudible]`` — a word or phrase lost on the line.
* ``--`` at the end of a turn — the speaker was cut off; the next speaker
  starts slightly before they finished.
"""

from __future__ import annotations

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


def format_offset(seconds: float) -> str:
    """Format a call offset or duration as ``HH:MM:SS``."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
