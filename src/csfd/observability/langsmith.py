"""Opt-in LangSmith tracing setup. No-op unless an API key is provided."""

from __future__ import annotations

import os


def maybe_enable_langsmith(*, api_key: str | None, project: str) -> bool:
    """Enable LangChain/LangSmith tracing if an API key is present.

    Returns True if tracing was enabled, False otherwise.
    """
    if not api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    return True
