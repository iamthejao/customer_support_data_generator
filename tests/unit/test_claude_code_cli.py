"""Tests for the Claude Code CLI provider.

The real ``claude`` binary is never invoked; the subprocess spawn function is
patched to return a fake process that emits prepared stdout/stderr bytes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from csfd.models import claude_code_cli as ccc
from csfd.models.claude_code_cli import ClaudeCodeCLIModel
from csfd.models.registry import build_llm
from csfd.settings import AgentLLMConfig


class _FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None: ...
    async def wait(self) -> None: ...


def _patch_spawn(monkeypatch: pytest.MonkeyPatch, responses: list[_FakeProc]) -> list[list[str]]:
    """Patch the async spawn to pop fake processes in order.

    Returns a list that will be filled with each call's argv for assertions.
    """
    calls: list[list[str]] = []
    queue = list(responses)

    async def fake_spawn(*argv: str, **_kwargs: Any) -> _FakeProc:
        calls.append(list(argv))
        return queue.pop(0)

    monkeypatch.setattr(ccc, "_spawn_subprocess", fake_spawn)
    return calls


def test_registry_dispatches_to_claude_code_cli() -> None:
    cfg = AgentLLMConfig(provider="claude_code_cli", model="claude-sonnet-4-5")
    llm = build_llm(cfg)
    assert isinstance(llm, ClaudeCodeCLIModel)
    assert llm.model == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_agenerate_parses_envelope_and_returns_ai_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = json.dumps(
        {"type": "result", "subtype": "success", "result": "hello world"}
    ).encode()
    calls = _patch_spawn(monkeypatch, [_FakeProc(stdout=envelope)])

    llm = ClaudeCodeCLIModel(model="claude-sonnet-4-5", binary="claude")
    result = await llm._agenerate([HumanMessage(content="hi")])

    assert result.generations[0].message.content == "hello world"
    argv = calls[0]
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "--model" in argv
    assert "claude-sonnet-4-5" in argv
    assert "--output-format" in argv
    assert "json" in argv


@pytest.mark.asyncio
async def test_with_structured_output_returns_validated_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Out(BaseModel):
        name: str
        count: int

    inner = {"name": "alpha", "count": 3}
    envelope = json.dumps({"result": json.dumps(inner)}).encode()
    _patch_spawn(monkeypatch, [_FakeProc(stdout=envelope)])

    llm = ClaudeCodeCLIModel()
    runnable = llm.with_structured_output(Out)
    parsed = await runnable.ainvoke("give me an Out")

    assert isinstance(parsed, Out)
    assert parsed.name == "alpha"
    assert parsed.count == 3


@pytest.mark.asyncio
async def test_with_structured_output_strips_code_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Out(BaseModel):
        x: int

    fenced = "```json\n" + json.dumps({"x": 7}) + "\n```"
    envelope = json.dumps({"result": fenced}).encode()
    _patch_spawn(monkeypatch, [_FakeProc(stdout=envelope)])

    parsed = await ClaudeCodeCLIModel().with_structured_output(Out).ainvoke("x?")
    assert isinstance(parsed, Out)
    assert parsed.x == 7


@pytest.mark.asyncio
async def test_with_structured_output_retries_once_on_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Out(BaseModel):
        x: int

    bad = json.dumps({"result": "not json at all"}).encode()
    good = json.dumps({"result": json.dumps({"x": 42})}).encode()
    calls = _patch_spawn(monkeypatch, [_FakeProc(stdout=bad), _FakeProc(stdout=good)])

    parsed = await ClaudeCodeCLIModel().with_structured_output(Out).ainvoke("x?")
    assert isinstance(parsed, Out)
    assert parsed.x == 42
    assert len(calls) == 2
    # The retry prompt is longer than the first (carries the reminder prefix).
    retry_prompt = calls[1][calls[1].index("-p") + 1]
    assert "previous response was not valid JSON" in retry_prompt


@pytest.mark.asyncio
async def test_non_zero_exit_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_spawn(
        monkeypatch,
        [_FakeProc(stdout=b"", stderr=b"oops something failed", returncode=1)],
    )
    llm = ClaudeCodeCLIModel()
    with pytest.raises(RuntimeError, match="claude CLI failed"):
        await llm._agenerate([HumanMessage(content="hi")])
