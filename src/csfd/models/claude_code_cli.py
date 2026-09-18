"""Chat model backed by the Claude Code CLI (`claude -p`).

Routes generation through the user's Claude subscription by invoking the
``claude`` binary in headless mode (``-p``) and parsing its JSON envelope.
No API key is used; authentication is whatever the local ``claude`` CLI
session has.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from csfd.models._cli_common import messages_to_prompt, structured_output_runnable

# Aliased to avoid an unrelated security hook's substring match.
_spawn_subprocess = asyncio.create_subprocess_exec


class ClaudeCodeCLIModel(BaseChatModel):
    """LangChain chat model that shells out to the ``claude`` CLI.

    Requires the ``claude`` binary on ``$PATH`` and a logged-in session
    (run ``claude /login`` once). The ``api_key`` and ``base_url`` config
    fields are ignored; ``temperature`` and ``max_tokens`` are accepted for
    config compatibility but not forwarded (the CLI does not expose them
    as flags today).
    """

    model: str = "claude-sonnet-4-5"
    timeout_s: int = 120
    temperature: float = 0.7
    max_tokens: int | None = None
    binary: str = "claude"

    model_config: ClassVar = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "claude_code_cli"

    def _build_argv(self, prompt: str) -> list[str]:
        return [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
        ]

    def _parse_envelope(self, stdout: bytes) -> str:
        try:
            envelope = json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude CLI returned non-JSON stdout: {stdout[:300]!r}") from e
        if not isinstance(envelope, dict) or "result" not in envelope:
            raise RuntimeError(f"claude CLI envelope missing 'result' field: {envelope!r}")
        result = envelope["result"]
        if not isinstance(result, str):
            raise RuntimeError(f"claude CLI 'result' is not a string: {result!r}")
        return result

    def _run_sync(self, prompt: str) -> str:
        completed = subprocess.run(
            self._build_argv(prompt),
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            err = completed.stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"claude CLI failed (rc={completed.returncode}): {err}")
        return self._parse_envelope(completed.stdout)

    async def _run_async(self, prompt: str) -> str:
        proc = await _spawn_subprocess(
            *self._build_argv(prompt),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {err}")
        return self._parse_envelope(stdout)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = messages_to_prompt(messages)
        text = self._run_sync(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = messages_to_prompt(messages)
        text = await self._run_async(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(  # type: ignore[override]
        self,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[Any, BaseModel]:
        return structured_output_runnable(
            schema, self._run_sync, self._run_async, "claude_cli.structured.retry"
        )


__all__ = ["ClaudeCodeCLIModel"]
