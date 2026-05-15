from csfd.ticket_types.definitions import TICKET_TYPE_METADATA, TicketType


def test_ticket_type_enum_values() -> None:
    assert TicketType.DOCS_REQUEST.value == "docs_request"
    assert TicketType.L1.value == "l1"
    assert TicketType.L2.value == "l2"
    assert TicketType.L3.value == "l3"


def test_all_types_have_metadata() -> None:
    for t in TicketType:
        meta = TICKET_TYPE_METADATA[t]
        assert meta.avg_turns >= 1
        assert meta.persona_label
        assert meta.requires_kb in (True, False)


def test_l3_does_not_require_kb() -> None:
    assert TICKET_TYPE_METADATA[TicketType.L3].requires_kb is False


def test_docs_l1_l2_require_kb() -> None:
    for t in (TicketType.DOCS_REQUEST, TicketType.L1, TicketType.L2):
        assert TICKET_TYPE_METADATA[t].requires_kb is True
