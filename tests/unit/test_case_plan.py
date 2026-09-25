"""Unit tests for case-plan parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from csfd.case_plan import CasePlan, CasePlanEntry, load_case_plan


def test_end_modes_imply_the_contact_count() -> None:
    assert CasePlanEntry(end_modes=["follow_up", "dropped"]).contacts == 3


def test_unknown_keys_and_empty_plans_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CasePlanEntry.model_validate({"chanel": "phone"})
    with pytest.raises(ValidationError):
        CasePlan(cases=[])


def test_problem_pins_resolve_by_index_or_id() -> None:
    plan = CasePlan(cases=[CasePlanEntry(problem=1), CasePlanEntry(problem="r:p:0000")])
    ids = ["r:p:0000", "r:p:0001"]
    assert [plan.problem_id(c, ids) for c in plan.cases] == ["r:p:0001", "r:p:0000"]
    with pytest.raises(ValueError, match="not in this run"):
        plan.problem_id(CasePlanEntry(problem="other"), ids)


def test_load_case_plan_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text(
        "cases:\n  - ticket_type: l2\n    problem_state: pending_part\n    contacts: 1\n",
        encoding="utf-8",
    )
    [entry] = load_case_plan(path).cases
    assert (entry.ticket_type, entry.problem_state, entry.contacts) == ("l2", "pending_part", 1)
