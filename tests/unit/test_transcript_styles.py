"""Speaker-label and header options of the plain-text transcript renderer."""

from datetime import UTC, datetime
from typing import Any

from csfd.storage.transcripts import Case, Contact, case_payload, render_transcript


def _case(channel: str = "phone", facts: dict[str, Any] | None = None) -> tuple[Case, Contact]:
    contact = Contact(
        sequence=1,
        count=2,
        channel=channel,
        reason="Furnace alarm",
        customer_name="Customer-standard-0001",
        agent_name="Agent-l2-0001",
        agent_role="L2 Support",
        resolution_uid="r",
        started_at=datetime(2026, 1, 22, 14, 6, 36, tzinfo=UTC),
        ended_at=datetime(2026, 1, 22, 14, 9, 0, tzinfo=UTC),
        duration_s=144.0,
        end_reason="agent_done",
        resolved=False,
        quality_flag=None,
        turns=[
            {"speaker": "agent", "content": "Kalvora support.", "start_s": 0.0},
            {"speaker": "customer", "content": "Our CF-600 alarms.", "start_s": 6.0},
        ],
    )
    case = Case(
        case_id="429417d7-9b94-4f08:000001",
        run_id="429417d7-9b94-4f08",
        slot_index=1,
        problem_id="p",
        ticket_type="l2",
        customer_tier="standard",
        customer_tone="neutral",
        channel=channel,
        customer_name="Customer-standard-0001",
        company_name="Kalvora",
        facts=facts,
        contacts=[contact],
    )
    return case, contact


def _body(text: str) -> list[str]:
    return text.split("\n\n", 1)[1].splitlines()


def test_defaults_keep_the_csfd_layout() -> None:
    case, contact = _case(facts={"caller_role": "maintenance technician"})
    text = render_transcript(case, contact)
    assert text.startswith("CALL TRANSCRIPT\ncase_id: 429417d7-9b94-4f08:000001\n")
    assert "channel: phone (inbound)" in text
    assert _body(text) == [
        "[00:00:00] AGENT: Kalvora support.",
        "[00:00:06] CUSTOMER: Our CF-600 alarms.",
    ]


def test_title_speaker_style() -> None:
    case, contact = _case()
    lines = _body(render_transcript(case, contact, speaker_style="title"))
    assert lines == [
        "[00:00:00] Agent: Kalvora support.",
        "[00:00:06] Customer: Our CF-600 alarms.",
    ]


def test_role_speaker_style_uses_the_caller_role() -> None:
    case, contact = _case(facts={"caller_role": "maintenance technician"})
    lines = _body(render_transcript(case, contact, speaker_style="role", timestamps=False))
    assert lines == [
        "Agent: Kalvora support.",
        "Caller (maintenance technician): Our CF-600 alarms.",
    ]


def test_role_speaker_style_without_facts_falls_back_to_caller() -> None:
    case, contact = _case()
    lines = _body(render_transcript(case, contact, speaker_style="role", timestamps=False))
    assert lines[1] == "Caller: Our CF-600 alarms."


def test_wissant_header_is_a_metadata_block() -> None:
    case, contact = _case()
    text = render_transcript(case, contact, header_style="wissant", speaker_style="title")
    assert text.splitlines()[:4] == [
        "call_id: CSFD-429417D7-1-1",
        "call_date: 2026-01-22",
        "---",
        "[00:00:00] Agent: Kalvora support.",
    ]
    assert "phone (inbound)" not in text and "duration" not in text


def test_no_header() -> None:
    case, contact = _case()
    text = render_transcript(case, contact, header_style="none")
    assert text.splitlines()[0] == "[00:00:00] AGENT: Kalvora support."


def test_email_styles_touch_only_the_header() -> None:
    case, contact = _case(channel="email", facts={"caller_role": "plant engineer"})
    default = render_transcript(case, contact)
    styled = render_transcript(case, contact, speaker_style="role", header_style="wissant")
    assert styled.splitlines()[:3] == [
        "thread_id: CSFD-429417D7-1-1",
        "thread_date: 2026-01-22",
        "---",
    ]
    assert styled.split("---\n", 1)[1] == default.split("\n\n", 1)[1]


def test_case_payload_carries_the_facts() -> None:
    facts = {"asset_model": "CF-600", "asset_serial": "CF600-0862-YR"}
    case, _ = _case(facts=facts)
    assert case_payload(case)["case_facts"] == facts
