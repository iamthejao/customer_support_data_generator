"""OpenAICompatEmbedder: mock the openai SDK, assert call shape + Matryoshka truncate."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from csfd.embeddings.client import OpenAICompatEmbedder


def _mock_openai_response(vectors: list[list[float]]) -> Any:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=v) for v in vectors]
    return resp


def test_embed_passes_model_and_input_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    create = AsyncMock(return_value=_mock_openai_response([[1.0, 0.0, 0.0, 0.0]]))
    fake_client = MagicMock()
    fake_client.embeddings.create = create

    def _factory(**kw: Any) -> Any:
        assert kw["base_url"] == "http://localhost:11434/v1"
        assert kw["api_key"] == "ollama"
        return fake_client

    monkeypatch.setattr("csfd.embeddings.client.AsyncOpenAI", _factory)

    embedder = OpenAICompatEmbedder(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="embeddinggemma:300m",
        dim=None,
        timeout_s=30,
    )
    out = asyncio.run(embedder.embed(["hello"]))
    assert out == [[1.0, 0.0, 0.0, 0.0]]
    create.assert_awaited_once_with(model="embeddinggemma:300m", input=["hello"])


def test_embed_truncates_to_dim_and_renormalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    create = AsyncMock(return_value=_mock_openai_response([[3.0, 4.0, 99.0, 99.0]]))
    fake_client = MagicMock()
    fake_client.embeddings.create = create
    monkeypatch.setattr("csfd.embeddings.client.AsyncOpenAI", lambda **kw: fake_client)

    embedder = OpenAICompatEmbedder(
        base_url="x",
        api_key="x",
        model="m",
        dim=2,
        timeout_s=30,
    )
    out = asyncio.run(embedder.embed(["t"]))
    assert len(out[0]) == 2
    assert out[0][0] == pytest.approx(0.6)
    assert out[0][1] == pytest.approx(0.8)


def test_embed_passes_through_when_dim_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    create = AsyncMock(return_value=_mock_openai_response([[0.1, 0.2, 0.3, 0.4]]))
    fake_client = MagicMock()
    fake_client.embeddings.create = create
    monkeypatch.setattr("csfd.embeddings.client.AsyncOpenAI", lambda **kw: fake_client)

    embedder = OpenAICompatEmbedder(
        base_url="x",
        api_key="x",
        model="m",
        dim=None,
        timeout_s=30,
    )
    out = asyncio.run(embedder.embed(["t"]))
    assert out == [[0.1, 0.2, 0.3, 0.4]]
