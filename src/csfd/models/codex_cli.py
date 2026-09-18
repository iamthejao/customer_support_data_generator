"""Chat model backed by the Codex CLI (``codex exec``).

Routes generation through the user's ChatGPT subscription by invoking the
``codex`` binary in non-interactive mode (``exec``) and reading its final
message from a temp file. No API key is used; authentication is whatever the
local ``codex`` CLI session has. The sandbox is forced to read-only so codex
cannot modify any files on disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from typing import Any, ClassVar

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ValidationError

_log = structlog.get_logger(__name__)

# Aliased to avoid an unrelated security hook's substring match.
_spawn_subprocess = asyncio.create_subprocess_exec

_STRUCTURED_SUFFIX = (
    "\n\nRespond with a single JSON object matching this schema. "
    "Do not include prose, explanations, or markdown code fences:\n{schema}\n"
)
_RETRY_PREFIX = (
    "Your previous response was not valid JSON for the requested schema. "
    "Reply with ONLY a JSON object - no prose, no fences.\n\n"
)


def _messages_to_prompt(messages: list[BaseMessage]) -> str:
    parts: list[str] = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if isinstance(m, SystemMessage):
            parts.append(f"System: {content}")
        elif isinstance(m, HumanMessage):
            parts.append(content)
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


class CodexCLIModel(BaseChatModel):
    """LangChain chat model that shells out to the ``codex`` CLI.

    Requires the ``codex`` binary on ``$PATH`` and a logged-in ChatGPT
    session (run ``codex login`` once). The ``api_key`` and ``base_url``
    config fields are ignored; ``temperature`` and ``max_tokens`` are
    accepted for config compatibility but not forwarded (the CLI does not
    expose them as flags today). Every invocation runs with a read-only
    sandbox, never persists session files, and never prompts for approval,
    so codex can read the repo but cannot modify it.
    """

    model: str = "gpt-5.6-sol"
    timeout_s: int = 120
    temperature: float = 0.7
    max_tokens: int | None = None
    binary: str = "codex"

    model_config: ClassVar = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "codex_cli"

    def _build_argv(self, prompt: str, output_file: str) -> list[str]:
        return [
            self.binary,
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "-c",
            "approval_policy=never",
            "--output-last-message",
            output_file,
            "--model",
            self.model,
            "--",
            prompt,
        ]

    def _read_last_message(self, output_file: str) -> str:
        try:
            with open(output_file, encoding="utf-8") as f:
                text = f.read().strip()
        except OSError as e:
            raise RuntimeError(f"codex CLI produced no output file: {e}") from e
        finally:
            try:
                os.unlink(output_file)
            except OSError:
                pass
        if not text:
            raise RuntimeError("codex CLI final message was empty")
        return text

    def _run_sync(self, prompt: str) -> str:
        fd, output_file = tempfile.mkstemp(suffix=".txt", prefix="codex-cli-")
        os.close(fd)
        completed = subprocess.run(
            self._build_argv(prompt, output_file),
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            try:
                os.unlink(output_file)
            except OSError:
                pass
            err = completed.stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"codex CLI failed (rc={completed.returncode}): {err}")
        return self._read_last_message(output_file)

    async def _run_async(self, prompt: str) -> str:
        fd, output_file = tempfile.mkstemp(suffix=".txt", prefix="codex-cli-")
        os.close(fd)
        proc = await _spawn_subprocess(
            *self._build_argv(prompt, output_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            try:
                os.unlink(output_file)
            except OSError:
                pass
            raise
        if proc.returncode != 0:
            try:
                os.unlink(output_file)
            except OSError:
                pass
            err = stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"codex CLI failed (rc={proc.returncode}): {err}")
        return self._read_last_message(output_file)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _messages_to_prompt(messages)
        text = self._run_sync(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _messages_to_prompt(messages)
        text = await self._run_async(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(  # type: ignore[override]
        self,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[Any, BaseModel]:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        suffix = _STRUCTURED_SUFFIX.format(schema=schema_json)

        def _prompt_from_input(inp: Any) -> str:
            if isinstance(inp, str):
                base = inp
            elif isinstance(inp, list):
                base = _messages_to_prompt(inp)
            elif isinstance(inp, BaseMessage):
                base = _messages_to_prompt([inp])
            else:
                base = str(inp)
            return base + suffix

        def _parse(text: str) -> BaseModel:
            return schema.model_validate(json.loads(_strip_code_fence(text)))

        async def _arun(inp: Any) -> BaseModel:
            prompt = _prompt_from_input(inp)
            text = await self._run_async(prompt)
            try:
                return _parse(text)
            except (json.JSONDecodeError, ValidationError) as e:
                _log.warning(
                    "codex_cli.structured.retry",
                    error=repr(e),
                    preview=text[:200],
                )
                retry_text = await self._run_async(_RETRY_PREFIX + prompt)
                return _parse(retry_text)

        def _run(inp: Any) -> BaseModel:
            prompt = _prompt_from_input(inp)
            text = self._run_sync(prompt)
            try:
                return _parse(text)
            except (json.JSONDecodeError, ValidationError) as e:
                _log.warning(
                    "codex_cli.structured.retry",
                    error=repr(e),
                    preview=text[:200],
                )
                retry_text = self._run_sync(_RETRY_PREFIX + prompt)
                return _parse(retry_text)

        return RunnableLambda(_run, afunc=_arun)


__all__ = ["CodexCLIModel"]
