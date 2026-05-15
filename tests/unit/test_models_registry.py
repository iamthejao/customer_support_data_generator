import pytest

from csfd.models.fake import FakeChatModel
from csfd.models.registry import build_llm
from csfd.settings import AgentLLMConfig


def test_build_anthropic_llm_constructs_client() -> None:
    cfg = AgentLLMConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.1)
    llm = build_llm(cfg)
    assert type(llm).__name__ == "ChatAnthropic"


def test_build_openai_compat_llm_uses_base_url() -> None:
    cfg = AgentLLMConfig(
        provider="openai_compat", model="llama3.1:8b",
        temperature=0.1, base_url="http://localhost:11434/v1",
        api_key="local",
    )
    llm = build_llm(cfg)
    assert type(llm).__name__ == "ChatOpenAI"


def test_fake_chat_model_returns_canned_response() -> None:
    llm = FakeChatModel(canned={"default": "hello"})
    result = llm.invoke("anything")
    assert "hello" in str(result.content)


def test_build_llm_unknown_provider_raises() -> None:
    cfg = AgentLLMConfig.model_construct(provider="nope", model="x", temperature=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_llm(cfg)
