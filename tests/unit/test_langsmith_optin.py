import os

import pytest

from csfd.observability.langsmith import maybe_enable_langsmith


def test_no_op_when_api_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    result = maybe_enable_langsmith(api_key=None, project="csfd")
    assert result is False
    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_sets_env_when_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    result = maybe_enable_langsmith(api_key="lsk_test", project="my-proj")
    assert result is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "my-proj"
    assert os.environ["LANGCHAIN_API_KEY"] == "lsk_test"
