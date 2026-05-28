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


def test_structured_seq_consumed_in_order() -> None:
    from pydantic import BaseModel

    from csfd.models.fake import FakeChatModel

    class S(BaseModel):
        v: int

    fake = FakeChatModel(structured_seq={S: [S(v=1), S(v=2)]})
    runnable = fake.with_structured_output(S)
    first = runnable.invoke("x")
    second = runnable.invoke("x")
    assert isinstance(first, S) and first.v == 1
    assert isinstance(second, S) and second.v == 2


def test_structured_seq_falls_back_to_structured_when_empty() -> None:
    from pydantic import BaseModel

    from csfd.models.fake import FakeChatModel

    class S(BaseModel):
        v: int

    fake = FakeChatModel(structured={S: S(v=9)}, structured_seq={S: []})
    out = fake.with_structured_output(S).invoke("x")
    assert isinstance(out, S) and out.v == 9
