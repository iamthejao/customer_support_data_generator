"""Content hashing helpers used for prompt versioning and dedup."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def short_hash(content: str, length: int = 12) -> str:
    """Stable short SHA-256 hex digest of `content` after stripping whitespace."""
    digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
    return digest[:length]


def stable_json_hash(obj: Any, length: int = 12) -> str:
    """Hash an object via canonical JSON encoding (sorted keys, separators)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return short_hash(payload, length=length)
