"""Export a run's conversations as plain-text transcripts grouped by case.

This is the text-first view of Phase 2 output, meant as direct input to
downstream consumers (e.g. a report pipeline that reads call transcripts)::

    <out>/<run_id>/transcripts/
        cases.jsonl                 # one line per case: metadata + utterances
        case_000001/
            case.json               # case metadata (no transcript text)
            call_01.txt             # phone channel; email_01.txt for email

Each ``.txt`` file is a ``key: value`` header, a blank line, then the
transcript. Ground truth (root cause, resolution hints) is deliberately not
written here; join ``case.json``'s ``problem_id`` against ``problems.jsonl``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from csfd.calls import HOLD_TAG, format_offset
from csfd.storage.db import Database
from csfd.ticket_types.definitions import TICKET_TYPE_METADATA, TicketType

_CASES_SQL = """
SELECT l.ticket_uid, l.slot_index, l.problem_id, l.ticket_type,
       l.customer_tier, l.customer_tone,
       ir.customer_name, ir.channel, ir.subject,
       res.resolution_uid, res.turns_json, res.resolved, res.quality_flag,
       res.agent_name, res.end_reason, res.started_at, res.ended_at, res.duration_s
FROM lineage l
JOIN incoming_requests ir ON ir.id = l.incoming_request_id
JOIN resolutions res ON res.incoming_request_id = ir.id
WHERE l.run_id = ?
ORDER BY l.slot_index
"""

_FILE_NOUN = {"phone": "call", "email": "email"}


@dataclass(slots=True)
class Contact:
    """One call (or email exchange) within a case."""

    sequence: int
    channel: str
    reason: str
    customer_name: str
    agent_name: str | None
    agent_role: str
    resolution_uid: str
    started_at: datetime | None
    ended_at: datetime | None
    duration_s: float | None
    end_reason: str | None
    resolved: bool
    quality_flag: str | None
    turns: list[dict[str, Any]]
    count: int = 1

    @property
    def file_name(self) -> str:
        return f"{_FILE_NOUN.get(self.channel, 'contact')}_{self.sequence:02d}.txt"

    @property
    def outcome(self) -> str | None:
        return self.turns[-1].get("done_reason") if self.turns else None


@dataclass(slots=True)
class Case:
    """All contacts that belong to one allocation slot."""

    case_id: str
    run_id: str
    slot_index: int
    problem_id: str
    ticket_type: str
    customer_tier: str
    customer_tone: str
    channel: str
    customer_name: str
    contacts: list[Contact] = field(default_factory=list)

    @property
    def dir_name(self) -> str:
        return f"case_{self.slot_index:06d}"


def _parse_ts(v: str | None) -> datetime | None:
    return datetime.fromisoformat(v) if v else None


def _agent_role(ticket_type: str) -> str:
    try:
        return TICKET_TYPE_METADATA[TicketType(ticket_type)].persona_label
    except ValueError:
        return ticket_type


def load_cases(db: Database, run_id: str) -> list[Case]:
    """Read every committed case of a run, contacts ordered by sequence."""
    with db.connect() as conn:
        rows = conn.execute(_CASES_SQL, (run_id,)).fetchall()
    cases: dict[str, Case] = {}
    for r in rows:
        case = cases.get(r["ticket_uid"])
        if case is None:
            case = cases[r["ticket_uid"]] = Case(
                case_id=r["ticket_uid"],
                run_id=run_id,
                slot_index=int(r["slot_index"]),
                problem_id=r["problem_id"],
                ticket_type=r["ticket_type"],
                customer_tier=r["customer_tier"],
                customer_tone=r["customer_tone"],
                channel=r["channel"],
                customer_name=r["customer_name"],
            )
        case.contacts.append(
            Contact(
                sequence=len(case.contacts) + 1,
                channel=r["channel"],
                reason=r["subject"],
                customer_name=r["customer_name"],
                agent_name=r["agent_name"],
                agent_role=_agent_role(r["ticket_type"]),
                resolution_uid=r["resolution_uid"],
                started_at=_parse_ts(r["started_at"]),
                ended_at=_parse_ts(r["ended_at"]),
                duration_s=r["duration_s"],
                end_reason=r["end_reason"],
                resolved=bool(r["resolved"]),
                quality_flag=r["quality_flag"],
                turns=json.loads(r["turns_json"]),
            )
        )
    for case in cases.values():
        for c in case.contacts:
            c.count = len(case.contacts)
    return list(cases.values())


def _gap_s(case: Case, contact: Contact) -> float | None:
    """Seconds between the previous contact's end (or start) and this one's start."""
    if contact.sequence == 1 or contact.started_at is None:
        return None
    prev = case.contacts[contact.sequence - 2]
    prev_end = prev.ended_at or prev.started_at
    if prev_end is None:
        return None
    return round((contact.started_at - prev_end).total_seconds(), 1)


def _format_gap(seconds: float) -> str:
    minutes = int(seconds // 60)
    days, rem = divmod(minutes, 24 * 60)
    hours, mins = divmod(rem, 60)
    parts = [f"{days}d"] if days else []
    parts += [f"{hours}h", f"{mins:02d}m"] if (days or hours) else [f"{mins}m"]
    return " ".join(parts)


def _header(case: Case, contact: Contact) -> list[str]:
    lines = [
        "CALL TRANSCRIPT" if contact.channel == "phone" else "EMAIL EXCHANGE",
        f"case_id: {case.case_id}",
    ]
    noun = _FILE_NOUN.get(contact.channel, "contact")
    lines.append(f"{noun}: {contact.sequence} of {contact.count}")
    lines.append("channel: phone (inbound)" if contact.channel == "phone" else "channel: email")
    if contact.started_at is not None:
        lines.append(f"started_at: {contact.started_at.isoformat()}")
    if contact.ended_at is not None:
        lines.append(f"ended_at: {contact.ended_at.isoformat()}")
    if contact.duration_s is not None:
        lines.append(f"duration: {format_offset(contact.duration_s)}")
    gap = _gap_s(case, contact)
    if gap is not None:
        lines.append(f"since_previous_{noun}: {_format_gap(gap)}")
    if contact.channel != "phone":
        lines.append(f"subject: {contact.reason}")
    role = "caller" if contact.channel == "phone" else "customer"
    lines.append(f"{role}: {contact.customer_name} ({case.customer_tier} tier)")
    if contact.agent_name:
        lines.append(f"agent: {contact.agent_name} ({contact.agent_role})")
    return lines


def _phone_lines(contact: Contact, *, timestamps: bool) -> list[str]:
    lines: list[str] = []
    for turn in contact.turns:
        text = str(turn["content"]).strip()
        start = float(turn.get("start_s", 0.0))
        hold = float(turn.get("hold_s", 0.0))
        if text.lower().startswith(HOLD_TAG):
            text = text[len(HOLD_TAG) :].lstrip()
            if hold:
                event = f"(caller on hold, {format_offset(hold)})"
                lines.append(f"[{format_offset(start - hold)}] {event}" if timestamps else event)
        speaker = str(turn["speaker"]).upper()
        prefix = f"[{format_offset(start)}] " if timestamps and "start_s" in turn else ""
        lines.append(f"{prefix}{speaker}: {text}")
    return lines


def _email_lines(contact: Contact) -> list[str]:
    lines: list[str] = []
    for turn in contact.turns:
        if lines:
            lines.append("")
        lines.append(f"--- {str(turn['speaker']).upper()} ---")
        lines.append(str(turn["content"]).strip())
    return lines


def render_transcript(case: Case, contact: Contact, *, timestamps: bool = True) -> str:
    """Render one contact as header + blank line + transcript text."""
    body = (
        _phone_lines(contact, timestamps=timestamps)
        if contact.channel == "phone"
        else _email_lines(contact)
    )
    return "\n".join([*_header(case, contact), "", *body]) + "\n"


def _iso(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


def _contact_payload(case: Case, contact: Contact, *, utterances: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sequence": contact.sequence,
        "file": contact.file_name,
        "resolution_uid": contact.resolution_uid,
        "reason": contact.reason,
        "agent_name": contact.agent_name,
        "started_at": _iso(contact.started_at),
        "ended_at": _iso(contact.ended_at),
        "duration_s": contact.duration_s,
        "gap_since_previous_s": _gap_s(case, contact),
        "end_reason": contact.end_reason,
        "outcome": contact.outcome,
        "resolved": contact.resolved,
        "quality_flag": contact.quality_flag,
        "turn_count": len(contact.turns),
    }
    if utterances:
        payload["utterances"] = [
            {
                "speaker": t["speaker"],
                "text": t["content"],
                **{k: t[k] for k in ("start_s", "end_s", "hold_s") if k in t},
            }
            for t in contact.turns
        ]
    return payload


def case_payload(case: Case, *, utterances: bool = False) -> dict[str, Any]:
    """Structured case record; ``utterances=True`` adds each contact's turns."""
    final = case.contacts[-1] if case.contacts else None
    return {
        "case_id": case.case_id,
        "run_id": case.run_id,
        "slot_index": case.slot_index,
        "problem_id": case.problem_id,
        "ticket_type": case.ticket_type,
        "customer_tier": case.customer_tier,
        "customer_tone": case.customer_tone,
        "customer_name": case.customer_name,
        "channel": case.channel,
        "contact_count": len(case.contacts),
        "resolved": final.resolved if final else False,
        "contacts": [_contact_payload(case, c, utterances=utterances) for c in case.contacts],
    }


def export_run_transcripts(
    db: Database, run_id: str, *, out_dir: Path, timestamps: bool = True
) -> list[Path]:
    """Write per-case transcript folders plus ``cases.jsonl``. Returns the files written."""
    root = out_dir / run_id / "transcripts"
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cases = load_cases(db, run_id)
    index_path = root / "cases.jsonl"
    with index_path.open("w", encoding="utf-8") as index:
        for case in cases:
            case_dir = root / case.dir_name
            case_dir.mkdir(exist_ok=True)
            for contact in case.contacts:
                path = case_dir / contact.file_name
                path.write_text(
                    render_transcript(case, contact, timestamps=timestamps), encoding="utf-8"
                )
                written.append(path)
            meta_path = case_dir / "case.json"
            meta_path.write_text(
                json.dumps(case_payload(case), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            written.append(meta_path)
            index.write(json.dumps(case_payload(case, utterances=True), ensure_ascii=False))
            index.write("\n")
    written.append(index_path)
    return written
