import pytest

from csfd.utils.rng import derive_rng, weighted_choice


def test_derive_rng_is_deterministic_with_same_seed_and_label() -> None:
    a = derive_rng(seed=42, label="phase1")
    b = derive_rng(seed=42, label="phase1")
    assert a.random() == b.random()


def test_derive_rng_diverges_on_different_label() -> None:
    a = derive_rng(seed=42, label="phase1")
    b = derive_rng(seed=42, label="phase2")
    assert a.random() != b.random()


def test_weighted_choice_respects_weights_over_many_trials() -> None:
    rng = derive_rng(seed=123, label="weights")
    weights = {"a": 0.8, "b": 0.2}
    counts = {"a": 0, "b": 0}
    for _ in range(5000):
        counts[weighted_choice(weights, rng)] += 1
    assert counts["a"] > 3500
    assert counts["b"] < 1500


def test_weighted_choice_rejects_empty_weights() -> None:
    rng = derive_rng(seed=0, label="x")
    with pytest.raises(ValueError):
        weighted_choice({}, rng)
