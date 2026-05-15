"""Per-run token + USD budget tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from csfd.errors import BudgetExceededError


@dataclass(slots=True)
class BudgetTracker:
    max_tokens: int
    max_usd: float
    _tokens_used: int = field(default=0, init=False)
    _usd_used: float = field(default=0.0, init=False)

    def record(self, *, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        self._tokens_used += tokens_in + tokens_out
        self._usd_used += cost_usd
        self._check()

    def _check(self) -> None:
        if self._tokens_used > self.max_tokens or self._usd_used > self.max_usd:
            raise BudgetExceededError(stats=self.stats())

    def stats(self) -> dict[str, Any]:
        return {
            "tokens": self._tokens_used,
            "usd": round(self._usd_used, 6),
            "max_tokens": self.max_tokens,
            "max_usd": self.max_usd,
        }
