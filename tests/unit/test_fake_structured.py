import pytest
from pydantic import BaseModel

from csfd.models.fake import FakeChatModel


class _Out(BaseModel):
    x: int
    label: str


def test_fake_with_structured_output_returns_typed_instance() -> None:
    canned = _Out(x=42, label="hi")
    llm = FakeChatModel(structured={_Out: canned})
    runnable = llm.with_structured_output(_Out)
    result = runnable.invoke("anything")
    assert isinstance(result, _Out)
    assert result.x == 42
    assert result.label == "hi"


def test_fake_with_structured_output_raises_when_no_canned() -> None:
    llm = FakeChatModel()
    runnable = llm.with_structured_output(_Out)
    with pytest.raises(KeyError):
        runnable.invoke("anything")
