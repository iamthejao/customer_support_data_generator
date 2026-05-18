import pytest

from csfd.utils.rng import derive_rng, largest_remainder, weighted_choice


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


def test_largest_remainder_sums_to_total() -> None:
    result = largest_remainder({"a": 0.5, "b": 0.3, "c": 0.2}, total=100)
    assert sum(result.values()) == 100
    assert result == {"a": 50, "b": 30, "c": 20}


def test_largest_remainder_handles_rounding() -> None:
    # 1/3, 1/3, 1/3 over 10 -> 4, 3, 3 (input-order tiebreak gives "a" the extra)
    result = largest_remainder({"a": 1, "b": 1, "c": 1}, total=10)
    assert sum(result.values()) == 10
    assert result == {"a": 4, "b": 3, "c": 3}


def test_largest_remainder_total_zero() -> None:
    assert largest_remainder({"a": 0.5, "b": 0.5}, total=0) == {"a": 0, "b": 0}


def test_largest_remainder_single_bucket() -> None:
    assert largest_remainder({"only": 1.0}, total=7) == {"only": 7}


def test_largest_remainder_input_order_tiebreak() -> None:
    # Equal weights, equal remainders -> earlier keys win
    result = largest_remainder({"x": 1, "y": 1}, total=3)
    assert result == {"x": 2, "y": 1}


def test_largest_remainder_rejects_negative_total() -> None:
    with pytest.raises(ValueError):
        largest_remainder({"a": 1.0}, total=-1)


def test_largest_remainder_rejects_zero_weights() -> None:
    with pytest.raises(ValueError):
        largest_remainder({"a": 0.0, "b": 0.0}, total=5)


def test_largest_remainder_proportions_need_not_sum_to_one() -> None:
    # 5:3:2 ratio -> same as 0.5/0.3/0.2 over 100
    assert largest_remainder({"a": 5, "b": 3, "c": 2}, total=100) == {"a": 50, "b": 30, "c": 20}
