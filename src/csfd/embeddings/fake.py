"""Deterministic in-memory embedder for tests. Never touches the network."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping


class FakeEmbedder:
    """Embeds text by hashing it and deriving a unit vector of length `dim`.

    Pass `force={"some text": (1.0, 0.0, ...)}` to override specific inputs --
    useful for integration tests that need to assert specific cosine scores.
    """

    def __init__(
        self,
        *,
        dim: int = 16,
        force: Mapping[str, tuple[float, ...]] | None = None,
    ) -> None:
        self.dim = dim
        self._force = dict(force or {})

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        if text in self._force:
            forced = self._force[text]
            if len(forced) != self.dim:
                raise ValueError(
                    f"forced vector for {text!r} has dim {len(forced)}, expected {self.dim}"
                )
            return list(forced)

        # Hash-derive `dim` floats and L2-normalize. Always non-zero norm.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        floats: list[float] = []
        i = 0
        while len(floats) < self.dim:
            if i + 4 > len(h):
                h = hashlib.sha256(h).digest()
                i = 0
            (u,) = struct.unpack("<I", h[i : i + 4])
            i += 4
            # Map [0, 2^32) -> [-1, 1).
            floats.append((u / 2_147_483_647.5) - 1.0)
        norm = math.sqrt(sum(x * x for x in floats)) or 1.0
        return [x / norm for x in floats]
