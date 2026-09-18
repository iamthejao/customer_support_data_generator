"""Tests for the Codex CLI provider.

The real ``codex`` binary is never invoked; the subprocess spawn function is
patched to return a fake process, and the ``--output-last-message`` file it
would have written is pre-populated before the model reads it back.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from csfd.models import codex_cli as cc
from csfd.models.codex_cli import CodexCLIModel
from csfd.models.registry import build_llm
from csfd.settings import AgentLLMConfig


class _FakeProc:
    def __init__(self, stderr: bytes = b"", returncode: int = 0) -> None:
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", self._stderr

    def kill(self) -> None: ...
    async def wait(self) -> None: ...


def _patch_spawn(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[_FakeProc],
    last_messages: list[str | None],
) -> list[list[str]]:
    """Patch the async spawn to pop fake processes in order.

    Before returning each fake process, writes ``last_messages[i]`` (if not
    None) to the ``--output-last-message`` path found in that call's argv,
    mimicking what the real ``codex exec`` binary would have written.
    Returns a list that will be filled with each call's argv for assertions.
    """
    calls: list[list[str]] = []
    proc_queue = list(responses)
    msg_queue = list(last_messages)

    async def fake_spawn(*argv: str, **kwargs: Any) -> _FakeProc:
        assert kwargs.get("stdin") is not None, "codex exec must not inherit stdin"
        calls.append(list(argv))
        output_file = argv[argv.index("--output-last-message") + 1]
        message = msg_queue.pop(0)
        if message is not None:
            with open(output_file, "w", encoding="utf-8") as f:  # noqa: ASYNC230
                f.write(message)
        return proc_queue.pop(0)

    monkeypatch.setattr(cc, "_spawn_subprocess", fake_spawn)
    return calls


def test_registry_dispatches_to_codex_cli() -> None:
    cfg = AgentLLMConfig(provider="codex_cli", model="gpt-5.6-sol")
    llm = build_llm(cfg)
    assert isinstance(llm, CodexCLIModel)
    assert llm.model == "gpt-5.6-sol"


@pytest.mark.asyncio
async def test_agenerate_reads_last_message_and_returns_ai_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_spawn(monkeypatch, [_FakeProc()], ["hello world"])

    llm = CodexCLIModel(model="gpt-5.6-sol", binary="codex")
    result = await llm._agenerate([HumanMessage(content="hi")])

    assert result.generations[0].message.content == "hello world"
    argv = calls[0]
    assert argv[0] == "codex"
    assert argv[1] == "exec"
    assert "--sandbox" in argv
    assert "read-only" in argv
    assert "--ephemeral" in argv
    assert "--ignore-user-config" in argv
    assert "--model" in argv
    assert "gpt-5.6-sol" in argv
    assert "--output-last-message" in argv
    assert argv[-2] == "--"
    assert argv[-1] == "hi"


@pytest.mark.asyncio
async def test_with_structured_output_returns_validated_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Out(BaseModel):
        name: str
        count: int

    inner = json.dumps({"name": "alpha", "count": 3})
    _patch_spawn(monkeypatch, [_FakeProc()], [inner])

    llm = CodexCLIModel()
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
    _patch_spawn(monkeypatch, [_FakeProc()], [fenced])

    parsed = await CodexCLIModel().with_structured_output(Out).ainvoke("x?")
    assert isinstance(parsed, Out)
    assert parsed.x == 7


@pytest.mark.asyncio
async def test_with_structured_output_retries_once_on_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Out(BaseModel):
        x: int

    good = json.dumps({"x": 42})
    calls = _patch_spawn(
        monkeypatch,
        [_FakeProc(), _FakeProc()],
        ["not json at all", good],
    )

    parsed = await CodexCLIModel().with_structured_output(Out).ainvoke("x?")
    assert isinstance(parsed, Out)
    assert parsed.x == 42
    assert len(calls) == 2
    retry_prompt = calls[1][-1]
    assert "previous response was not valid JSON" in retry_prompt


@pytest.mark.asyncio
async def test_non_zero_exit_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_spawn(
        monkeypatch,
        [_FakeProc(stderr=b"oops something failed", returncode=1)],
        [None],
    )
    llm = CodexCLIModel()
    with pytest.raises(RuntimeError, match="codex CLI failed"):
        await llm._agenerate([HumanMessage(content="hi")])


@pytest.mark.asyncio
async def test_empty_last_message_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_spawn(monkeypatch, [_FakeProc()], [""])
    llm = CodexCLIModel()
    with pytest.raises(RuntimeError, match="empty"):
        await llm._agenerate([HumanMessage(content="hi")])


def test_sync_timeout_removes_output_file(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_run(argv: list[str], **kwargs: Any) -> None:
        seen.append(argv[argv.index("--output-last-message") + 1])
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        CodexCLIModel(timeout_s=1)._generate([HumanMessage(content="hi")])
    assert seen
    assert not os.path.exists(seen[0])
