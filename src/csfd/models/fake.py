"""In-process fake chat model for unit and integration tests."""

from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field


class FakeChatModel(BaseChatModel):
    canned: dict[str, str] = Field(default_factory=dict)
    call_log: list[Any] = Field(default_factory=list)
    structured: dict[type[BaseModel], BaseModel] = Field(default_factory=dict)

    model_config: ClassVar = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_log.append(messages)
        content = self.canned.get("default", "")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def with_structured_output(  # type: ignore[override]
        self,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[Any, BaseModel]:
        """Return a runnable that outputs a canned structured response.

        Args:
            schema: The Pydantic model class to return.
            **kwargs: Additional arguments (ignored).

        Returns:
            A Runnable that returns the canned structured output for the schema.

        Raises:
            KeyError: If no canned output is configured for the schema.
        """
        canned = self.structured

        def _return_canned(_inputs: Any) -> BaseModel:
            if schema not in canned:
                raise KeyError(
                    f"FakeChatModel has no canned structured output for {schema.__name__}"
                )
            return canned[schema]

        return RunnableLambda(_return_canned)
