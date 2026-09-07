"""Deterministic retrieval metrics: hit-rate@k, MRR, nDCG@k.

No model is involved in any of these, which makes them the numbers to trust
most in the whole harness. They are computed from a ranked list of chunk IDs
against the golden set's ``gold_chunks``.

The single most important decision in this module
-------------------------------------------------
When a question has **no** gold chunks - which is exactly the unanswerable
slice of the golden set, ~12% of MultiHop-RAG - these metrics are *undefined*,
not zero. They return ``None``.

Scoring them as 0 would be a silent, systematic corruption: every arm would
take a zero on every unanswerable question, dragging all averages down by the
same amount, compressing the differences between arms, and making the
benchmark less able to detect the effect it exists to measure. Worse, it would
punish a system for correctly retrieving nothing when there was nothing to
retrieve. ``None`` propagates to the result row and those questions are
excluded from retrieval-metric aggregates - they are scored on abstention
instead.

This is the retrieval-side instance of the harness rule that "no answer" and
"negative answer" must never share a label.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

__all__ = ["dcg", "hit_rate_at_k", "mrr", "ndcg_at_k", "reciprocal_rank"]


def _validated(
    retrieved: Sequence[str], gold: Iterable[str], k: int | None
) -> tuple[list[str], set[str]] | None:
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    gold_set = set(gold)
    if not gold_set:
        return None
    ranked = list(retrieved) if k is None else list(retrieved)[:k]
    return ranked, gold_set


def hit_rate_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float | None:
    """1.0 if any gold chunk appears in the top ``k``, else 0.0.

    Returns ``None`` when there are no gold chunks (see module docstring).
    """
    validated = _validated(retrieved, gold, k)
    if validated is None:
        return None
    ranked, gold_set = validated
    return 1.0 if any(chunk_id in gold_set for chunk_id in ranked) else 0.0


def reciprocal_rank(
    retrieved: Sequence[str], gold: Iterable[str], k: int | None = None
) -> float | None:
    """``1 / rank`` of the first gold chunk, 0.0 if none is present.

    Only the *first* correct result counts - that is what distinguishes MRR
    from hit-rate.
    """
    validated = _validated(retrieved, gold, k)
    if validated is None:
        return None
    ranked, gold_set = validated
    for index, chunk_id in enumerate(ranked, start=1):
        if chunk_id in gold_set:
            return 1.0 / index
    return 0.0


def mrr(
    results: Iterable[tuple[Sequence[str], Iterable[str]]],
    k: int | None = None,
) -> float | None:
    """Mean reciprocal rank over many queries.

    Queries with no gold chunks are excluded rather than counted as zero. If
    every query is excluded, returns ``None``.
    """
    scores = [
        score
        for retrieved, gold in results
        if (score := reciprocal_rank(retrieved, gold, k)) is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain over a ranked list of relevance values.

    Uses the standard ``rel_i / log2(i + 1)`` with 1-indexed ranks, so the
    first position has discount ``log2(2) == 1``.
    """
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int = 10) -> float | None:
    """Normalised DCG at ``k`` with binary relevance.

    The ideal ranking places every gold chunk first, but no more of them than
    fit in ``k`` - so a query with 20 gold chunks scored at k=10 can still
    reach 1.0. Computing the ideal over all gold chunks instead would make a
    perfect ranking score below 1.0 and quietly penalise questions that happen
    to have many correct answers.

    Duplicate chunk IDs in ``retrieved`` are counted once: a retriever cannot
    earn extra credit by returning the same passage twice.
    """
    validated = _validated(retrieved, gold, k)
    if validated is None:
        return None
    ranked, gold_set = validated

    seen: set[str] = set()
    relevances: list[float] = []
    for chunk_id in ranked:
        if chunk_id in seen:
            relevances.append(0.0)
            continue
        seen.add(chunk_id)
        relevances.append(1.0 if chunk_id in gold_set else 0.0)

    ideal_hits = min(len(gold_set), k)
    ideal = dcg([1.0] * ideal_hits)
    if ideal == 0:
        return None
    return dcg(relevances) / ideal
