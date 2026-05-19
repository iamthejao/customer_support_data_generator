import math

import pytest

from csfd.embeddings.dedup import cosine, find_duplicate


def test_cosine_parallel_is_one() -> None:
    assert cosine((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one() -> None:
    assert cosine((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)


def test_cosine_handles_non_unit_vectors() -> None:
    assert cosine((2.0, 0.0), (3.0, 0.0)) == pytest.approx(1.0)


def test_cosine_zero_vector_returns_zero() -> None:
    # Avoid div-by-zero — define cos(_, 0) = 0.
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0


def test_find_duplicate_returns_highest_above_threshold() -> None:
    existing = [
        ("a", (1.0, 0.0, 0.0)),
        ("b", (0.9, math.sqrt(1 - 0.9**2), 0.0)),
        ("c", (0.0, 1.0, 0.0)),
    ]
    hit = find_duplicate((1.0, 0.0, 0.0), existing, threshold=0.85)
    assert hit is not None
    assert hit[0] == "a"
    assert hit[1] == pytest.approx(1.0)


def test_find_duplicate_returns_none_when_all_below() -> None:
    existing = [("a", (1.0, 0.0)), ("b", (0.9, math.sqrt(1 - 0.9**2)))]
    assert find_duplicate((0.0, 1.0), existing, threshold=0.95) is None


def test_find_duplicate_tie_keeps_first_seen() -> None:
    existing = [("a", (1.0, 0.0)), ("b", (1.0, 0.0))]
    hit = find_duplicate((1.0, 0.0), existing, threshold=0.5)
    assert hit is not None and hit[0] == "a"
