"""OpenAI-compatible embedding client. Used against Ollama's /v1/embeddings.

For Ollama:
    base_url=http://localhost:11434/v1, api_key="ollama" (dummy), model="embeddinggemma:300m"
For real OpenAI:
    base_url=https://api.openai.com/v1, api_key=$OPENAI_API_KEY, model="text-embedding-3-small"
"""

from __future__ import annotations

import math
from typing import Protocol

from openai import AsyncOpenAI


class EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        dim: int | None,
        timeout_s: int,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "unused",
            timeout=float(timeout_s),
        )
        self._model = model
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        vectors: list[list[float]] = [list(item.embedding) for item in resp.data]
        if self._dim is not None:
            vectors = [self._truncate_and_renorm(v, self._dim) for v in vectors]
        return vectors

    @staticmethod
    def _truncate_and_renorm(vec: list[float], dim: int) -> list[float]:
        if len(vec) <= dim:
            return vec
        sliced = vec[:dim]
        norm = math.sqrt(sum(x * x for x in sliced))
        if norm == 0.0:
            return sliced
        return [x / norm for x in sliced]
