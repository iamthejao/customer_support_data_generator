"""Stdlib-only lexical (TF-IDF cosine) dedup for ProblemDraft items."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from csfd.phases.phase1_kb.state import ProblemDraft

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _tf(tokens: list[str]) -> dict[str, float]:
    n = len(tokens) or 1
    counts = Counter(tokens)
    return {t: c / n for t, c in counts.items()}


def _idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    n_docs = len(corpus_tokens) or 1
    df: Counter[str] = Counter()
    for toks in corpus_tokens:
        for t in set(toks):
            df[t] += 1
    return {t: math.log((1 + n_docs) / (1 + df_t)) + 1.0 for t, df_t in df.items()}


def _vector(tf_doc: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    return {t: tf_v * idf.get(t, 0.0) for t, tf_v in tf_doc.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a if t in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def lexical_dedup(
    problems: Iterable[ProblemDraft],
    *,
    threshold: float,
) -> list[ProblemDraft]:
    """Return a deduplicated list — drop problems whose TF-IDF cosine similarity
    to any earlier-kept problem is at or above `threshold`."""
    items = list(problems)
    if not items:
        return []
    docs = [_tokenize(f"{p.title} {p.description}") for p in items]
    idf = _idf(docs)
    vectors = [_vector(_tf(d), idf) for d in docs]

    keep_indices: list[int] = []
    for i, vi in enumerate(vectors):
        is_dup = False
        for j in keep_indices:
            if _cosine(vi, vectors[j]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep_indices.append(i)
    return [items[i] for i in keep_indices]
