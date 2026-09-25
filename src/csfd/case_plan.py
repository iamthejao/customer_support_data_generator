"""An explicit case plan: pin what each case should be instead of seed-hunting.

``csfd generate --plan cases.yaml`` reads a YAML file with one entry per case::

    cases:
      - problem: 0                 # index into the run's problems, or a problem id
        ticket_type: l2
        tier: enterprise
        tone: frustrated
        contacts: 2                # contacts in the case
        end_modes: [follow_up]     # how contacts 1..N-1 end (follow_up | dropped)
        problem_state: pending_visit
      - ticket_type: l1            # anything left out is filled from the proportions

The number of entries is the number of cases (``tickets.total``). The
allocation plan is built from the configured proportions first, as for any run;
each entry then overrides the dimensions it pins, so unpinned dimensions stay
deterministic. With ``--problems-from <run_id>`` the plan can pin problems of an
earlier run, so the same problem can be rendered again in another channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from csfd.outcomes import ProblemState
from csfd.ticket_types.definitions import TicketType

NonFinalEndMode = Literal["follow_up", "dropped"]


class CasePlanEntry(BaseModel):
    """One case; every field is optional and pins that dimension when set."""

    model_config = ConfigDict(extra="forbid")

    problem: int | str | None = None  # 0-based index into the run's problems, or a problem id
    ticket_type: TicketType | None = None
    tier: str | None = None
    tone: str | None = None
    contacts: int | None = Field(default=None, ge=1)
    end_modes: list[NonFinalEndMode] | None = None
    problem_state: ProblemState | None = None

    @model_validator(mode="after")
    def _end_modes_fit_contacts(self) -> CasePlanEntry:
        if self.end_modes is None:
            return self
        if self.contacts is None:
            self.contacts = len(self.end_modes) + 1
        elif len(self.end_modes) != self.contacts - 1:
            raise ValueError(
                f"end_modes lists how contacts 1..{self.contacts - 1} end, so it needs "
                f"{self.contacts - 1} entries, not {len(self.end_modes)}"
            )
        return self


class CasePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[CasePlanEntry] = Field(min_length=1)

    def problem_id(self, entry: CasePlanEntry, problem_ids: list[str]) -> str | None:
        """Resolve an entry's ``problem`` pin against the run's problems (in commit order)."""
        pin = entry.problem
        if pin is None:
            return None
        if isinstance(pin, int):
            if not 0 <= pin < len(problem_ids):
                raise ValueError(
                    f"case plan pins problem index {pin}, but the run has "
                    f"{len(problem_ids)} problem(s)"
                )
            return problem_ids[pin]
        if pin not in problem_ids:
            raise ValueError(f"case plan pins problem {pin!r}, which is not in this run")
        return pin

    def check_problems(self, problem_ids: list[str]) -> None:
        """Fail early when a pinned problem does not exist."""
        for entry in self.cases:
            self.problem_id(entry, problem_ids)


def load_case_plan(path: Path) -> CasePlan:
    """Read and validate a case-plan YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CasePlan.model_validate(data)
