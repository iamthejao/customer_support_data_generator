from csfd.utils.hashing import short_hash, stable_json_hash


def test_short_hash_is_deterministic_and_12_chars() -> None:
    h1 = short_hash("hello world")
    h2 = short_hash("hello world")
    assert h1 == h2
    assert len(h1) == 12


def test_short_hash_strips_leading_trailing_whitespace() -> None:
    assert short_hash("hello world") == short_hash("   hello world   ")


def test_short_hash_changes_on_content_change() -> None:
    assert short_hash("a") != short_hash("b")


def test_stable_json_hash_ignores_dict_ordering() -> None:
    a = {"k1": 1, "k2": [3, 4]}
    b = {"k2": [3, 4], "k1": 1}
    assert stable_json_hash(a) == stable_json_hash(b)
