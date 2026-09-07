from __future__ import annotations

import math

import pytest

from adaptive_retrieval.metrics.retrieval import (
    dcg,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    reciprocal_rank,
)

# The worked example from the harness documentation: gold chunks c12 and c88,
# returned at ranks 2 and 4 of five results.
RANKED = ["c05", "c12", "c40", "c88", "c07"]
GOLD = ["c12", "c88"]


# --------------------------------------------------------------------------
# the documented worked example must reproduce exactly
# --------------------------------------------------------------------------


def test_worked_example_hit_rate() -> None:
    assert hit_rate_at_k(RANKED, GOLD, 10) == 1.0


def test_worked_example_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKED, GOLD) == pytest.approx(0.5)


def test_worked_example_ndcg() -> None:
    # DCG = 1/log2(3) + 1/log2(5) = 0.63093 + 0.43068 = 1.06161
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, GOLD, 10) == pytest.approx(expected)
    assert ndcg_at_k(RANKED, GOLD, 10) == pytest.approx(0.6509, abs=1e-4)


# --------------------------------------------------------------------------
# undefined, not zero - the module's central decision
# --------------------------------------------------------------------------


def test_no_gold_chunks_returns_none_not_zero() -> None:
    """Unanswerable questions have no gold chunks. Scoring them 0 would drag
    every arm's average down equally and compress the differences we exist to
    measure."""
    assert hit_rate_at_k(RANKED, [], 10) is None
    assert reciprocal_rank(RANKED, []) is None
    assert ndcg_at_k(RANKED, [], 10) is None


def test_none_is_distinguishable_from_a_genuine_zero() -> None:
    miss = hit_rate_at_k(["x", "y"], GOLD, 10)
    undefined = hit_rate_at_k(["x", "y"], [], 10)
    assert miss == 0.0
    assert undefined is None
    assert miss is not None


def test_mrr_excludes_undefined_queries_rather_than_zeroing_them() -> None:
    results = [
        (RANKED, GOLD),  # rr = 0.5
        (RANKED, []),  # undefined, must be excluded
    ]
    assert mrr(results) == pytest.approx(0.5)


def test_mrr_of_all_undefined_is_none() -> None:
    assert mrr([(RANKED, []), (RANKED, [])]) is None


def test_mrr_averages_defined_queries() -> None:
    results = [
        (["a", "b"], ["a"]),  # 1.0
        (["a", "b"], ["b"]),  # 0.5
    ]
    assert mrr(results) == pytest.approx(0.75)


# --------------------------------------------------------------------------
# hit rate
# --------------------------------------------------------------------------


def test_hit_rate_respects_k() -> None:
    assert hit_rate_at_k(RANKED, ["c07"], 4) == 0.0
    assert hit_rate_at_k(RANKED, ["c07"], 5) == 1.0


def test_hit_rate_at_one() -> None:
    assert hit_rate_at_k(RANKED, GOLD, 1) == 0.0
    assert hit_rate_at_k(RANKED, ["c05"], 1) == 1.0


def test_k_larger_than_result_list_is_fine() -> None:
    assert hit_rate_at_k(RANKED, GOLD, 1000) == 1.0


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        hit_rate_at_k(RANKED, GOLD, k)


def test_empty_result_list_is_a_miss_not_undefined() -> None:
    """The retriever returning nothing is a genuine failure, distinct from the
    question having no correct answer."""
    assert hit_rate_at_k([], GOLD, 10) == 0.0
    assert ndcg_at_k([], GOLD, 10) == 0.0


# --------------------------------------------------------------------------
# nDCG
# --------------------------------------------------------------------------


def test_perfect_ranking_scores_one() -> None:
    assert ndcg_at_k(["c12", "c88", "x", "y"], GOLD, 10) == pytest.approx(1.0)


def test_ideal_is_capped_at_k() -> None:
    """A query with more gold chunks than k must still be able to reach 1.0."""
    gold = [f"g{i}" for i in range(20)]
    retrieved = gold[:5]
    assert ndcg_at_k(retrieved, gold, 5) == pytest.approx(1.0)


def test_reversed_ranking_scores_lower() -> None:
    good = ndcg_at_k(["c12", "c88"], GOLD, 10)
    bad = ndcg_at_k(["x", "y", "c12", "c88"], GOLD, 10)
    assert good is not None and bad is not None
    assert good > bad


def test_duplicate_chunks_earn_credit_only_once() -> None:
    """A retriever must not be able to inflate nDCG by repeating a hit."""
    honest = ndcg_at_k(["c12", "x", "y"], ["c12"], 10)
    cheating = ndcg_at_k(["c12", "c12", "c12"], ["c12"], 10)
    assert honest == pytest.approx(cheating)


def test_dcg_uses_one_indexed_log2_discount() -> None:
    assert dcg([1.0]) == pytest.approx(1.0)
    assert dcg([0.0, 1.0]) == pytest.approx(1 / math.log2(3))


def test_dcg_of_nothing_is_zero() -> None:
    assert dcg([]) == 0.0


def test_ndcg_is_monotonic_in_position() -> None:
    scores = [ndcg_at_k(["x"] * i + ["c12"], ["c12"], 10) for i in range(5)]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]
