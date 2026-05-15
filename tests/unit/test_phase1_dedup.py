from csfd.phases.phase1_kb.dedup import lexical_dedup
from csfd.phases.phase1_kb.state import ProblemDraft


def _p(title: str, desc: str = "") -> ProblemDraft:
    return ProblemDraft(title=title, description=desc, category="x", severity="low")


def test_lexical_dedup_removes_near_duplicates() -> None:
    items = [
        _p("Cannot log in to my account", "I get a 401 error"),
        _p("Cannot log into my account", "I get a 401 error"),
        _p("Billing overage charge", "Charged for usage above plan"),
    ]
    keep = lexical_dedup(items, threshold=0.77)
    titles = sorted(p.title for p in keep)
    assert "Billing overage charge" in titles
    login_count = sum(1 for p in keep if "log" in p.title.lower())
    assert login_count == 1


def test_lexical_dedup_keeps_distinct_items() -> None:
    items = [
        _p("Issue A", "first"),
        _p("Issue B", "second"),
        _p("Issue C", "third"),
    ]
    keep = lexical_dedup(items, threshold=0.77)
    assert len(keep) == 3


def test_lexical_dedup_empty_input() -> None:
    assert lexical_dedup([], threshold=0.77) == []
