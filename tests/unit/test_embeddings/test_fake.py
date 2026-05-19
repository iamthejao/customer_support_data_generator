import asyncio

from csfd.embeddings.fake import FakeEmbedder


def test_fake_embedder_is_deterministic_per_input() -> None:
    fe = FakeEmbedder(dim=8)
    v1 = asyncio.run(fe.embed(["hello"]))[0]
    v2 = asyncio.run(fe.embed(["hello"]))[0]
    assert v1 == v2
    assert len(v1) == 8


def test_fake_embedder_different_text_different_vector() -> None:
    fe = FakeEmbedder(dim=8)
    v1 = asyncio.run(fe.embed(["hello"]))[0]
    v2 = asyncio.run(fe.embed(["world"]))[0]
    assert v1 != v2


def test_fake_embedder_force_override_takes_precedence() -> None:
    forced = (1.0, 0.0, 0.0, 0.0)
    fe = FakeEmbedder(dim=4, force={"x": forced})
    v = asyncio.run(fe.embed(["x"]))[0]
    assert tuple(v) == forced


def test_fake_embedder_batch_preserves_order() -> None:
    fe = FakeEmbedder(dim=4)
    out = asyncio.run(fe.embed(["a", "b", "a"]))
    assert out[0] == out[2]
    assert out[0] != out[1]
