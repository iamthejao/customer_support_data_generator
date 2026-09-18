"""Prompt-building and structured-output helpers shared by the CLI-backed models."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ValidationError

_log = structlog.get_logger(__name__)

_STRUCTURED_SUFFIX = (
    "\n\nRespond with a single JSON object matching this schema. "
    "Do not include prose, explanations, or markdown code fences:\n{schema}\n"
)
_RETRY_PREFIX = (
    "Your previous response was not valid JSON for the requested schema. "
    "Reply with ONLY a JSON object - no prose, no fences.\n\n"
)


def messages_to_prompt(messages: list[BaseMessage]) -> str:
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


def structured_output_runnable(
    schema: type[BaseModel],
    run_sync: Callable[[str], str],
    run_async: Callable[[str], Awaitable[str]],
    log_event: str,
) -> Runnable[Any, BaseModel]:
    """Prompt for JSON matching ``schema``, parse it, and retry once on bad JSON."""
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    suffix = _STRUCTURED_SUFFIX.format(schema=schema_json)

    def _prompt_from_input(inp: Any) -> str:
        if isinstance(inp, str):
            base = inp
        elif isinstance(inp, list):
            base = messages_to_prompt(inp)
        elif isinstance(inp, BaseMessage):
            base = messages_to_prompt([inp])
        else:
            base = str(inp)
        return base + suffix

    def _parse(text: str) -> BaseModel:
        return schema.model_validate(json.loads(_strip_code_fence(text)))

    def _warn(e: Exception, text: str) -> None:
        _log.warning(log_event, error=repr(e), preview=text[:200])

    async def _arun(inp: Any) -> BaseModel:
        prompt = _prompt_from_input(inp)
        text = await run_async(prompt)
        try:
            return _parse(text)
        except (json.JSONDecodeError, ValidationError) as e:
            _warn(e, text)
            return _parse(await run_async(_RETRY_PREFIX + prompt))

    def _run(inp: Any) -> BaseModel:
        prompt = _prompt_from_input(inp)
        text = run_sync(prompt)
        try:
            return _parse(text)
        except (json.JSONDecodeError, ValidationError) as e:
            _warn(e, text)
            return _parse(run_sync(_RETRY_PREFIX + prompt))

    return RunnableLambda(_run, afunc=_arun)
