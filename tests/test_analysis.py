from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adaptive_retrieval.analysis import compare_arms, load_rows, oracle_gap, summarise


def _row(arm: str, qid: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": "r1",
        "arm": arm,
        "query_id": qid,
        "query_class": "multi_hop",
        "rep": 0,
        "served_model": "claude-opus-5",
        "ndcg_at_k": 0.5,
        "hit_rate_at_k": 1.0,
        "mrr": 0.5,
        "citation_recall": 1.0,
        "abstained": False,
        "should_abstain": False,
        "cost_usd": 0.01,
        "judge_cost_usd": 0.002,
        "retried": False,
        "latency_ms": {"total": 1000.0},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------


def test_summarises_each_arm() -> None:
    rows = [_row("A1", "Q1"), _row("A1", "Q2"), _row("A4", "Q1")]
    summaries = {s.arm: s for s in summarise(rows)}
    assert summaries["A1"].n_rows == 2
    assert summaries["A4"].n_rows == 1


def test_undefined_metrics_are_excluded_not_zeroed() -> None:
    """An unanswerable question has no retrieval score. Averaging it as zero
    would drag every arm down equally and compress the differences."""
    rows = [_row("A1", "Q1", ndcg_at_k=0.8), _row("A1", "Q2", ndcg_at_k=None)]
    summary = summarise(rows)[0]
    assert summary.ndcg == pytest.approx(0.8)
    assert summary.n_rows == 2
    assert summary.n_scored == 1


def test_abstention_accuracy_scores_both_directions() -> None:
    """A one-sided metric would reward always-abstain."""
    rows = [
        _row("A1", "Q1", abstained=True, should_abstain=True),  # correct
        _row("A1", "Q2", abstained=True, should_abstain=False),  # wrongly refused
    ]
    assert summarise(rows)[0].abstention_accuracy == pytest.approx(0.5)


def test_judge_cost_is_summarised_separately_from_arm_cost() -> None:
    summary = summarise([_row("A1", "Q1")])[0]
    assert summary.mean_cost_usd == pytest.approx(0.01)
    assert summary.mean_judge_cost_usd == pytest.approx(0.002)


def test_latency_percentiles() -> None:
    rows = [_row("A1", f"Q{i}", latency_ms={"total": float(i)}) for i in range(1, 101)]
    summary = summarise(rows)[0]
    assert summary.p50_latency_ms < summary.p95_latency_ms


def test_retry_rate_is_reported() -> None:
    rows = [_row("A6", "Q1", retried=True), _row("A6", "Q2", retried=False)]
    assert summarise(rows)[0].retry_rate == pytest.approx(0.5)


def test_can_slice_by_query_class() -> None:
    rows = [
        _row("A1", "Q1", query_class="single_hop", ndcg_at_k=0.9),
        _row("A1", "Q2", query_class="multi_hop", ndcg_at_k=0.1),
    ]
    assert summarise(rows, query_class="single_hop")[0].ndcg == pytest.approx(0.9)


def test_loads_rows_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(_row("A1", "Q1")) + "\n\n", encoding="utf-8")
    assert len(load_rows(path)) == 1


# --------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------


def test_pairs_are_aligned_on_question_and_rep() -> None:
    rows = [
        _row("A6", "Q1", ndcg_at_k=0.9),
        _row("A7", "Q1", ndcg_at_k=0.4),
        _row("A6", "Q2", ndcg_at_k=0.8),
        _row("A7", "Q2", ndcg_at_k=0.3),
    ]
    ((pair, result, _),) = compare_arms(rows, [("A6", "A7")], resamples=500)
    assert pair == ("A6", "A7")
    assert result.n_pairs == 2
    assert result.mean_difference == pytest.approx(0.5)


def test_unmatched_questions_are_dropped_from_the_pairing() -> None:
    """A6 answered a question A7 did not; the pair cannot include it."""
    rows = [
        _row("A6", "Q1", ndcg_at_k=0.9),
        _row("A7", "Q1", ndcg_at_k=0.4),
        _row("A6", "Q2", ndcg_at_k=0.8),
    ]
    ((_, result, _),) = compare_arms(rows, [("A6", "A7")], resamples=500)
    assert result.n_pairs == 1


def test_bonferroni_is_applied_across_the_family() -> None:
    rows = [
        _row(arm, f"Q{i}", ndcg_at_k=0.5 + (0.1 if arm == "A6" else 0.0))
        for arm in ("A6", "A7", "A4")
        for i in range(20)
    ]
    comparisons = compare_arms(rows, [("A6", "A7"), ("A6", "A4"), ("A7", "A4")], resamples=500)
    for _, result, adjusted in comparisons:
        assert adjusted >= result.p_value


def test_comparison_of_a_missing_arm_is_skipped() -> None:
    assert compare_arms([_row("A6", "Q1")], [("A6", "A99")], resamples=100) == []


# --------------------------------------------------------------------------
# the oracle gap
# --------------------------------------------------------------------------


def test_oracle_picks_the_best_arm_per_question() -> None:
    rows = [
        # A1 is better on Q1, A5 is better on Q2. No fixed arm wins both.
        _row("A1", "Q1", ndcg_at_k=0.9),
        _row("A5", "Q1", ndcg_at_k=0.1),
        _row("A1", "Q2", ndcg_at_k=0.1),
        _row("A5", "Q2", ndcg_at_k=0.9),
        # The router got Q1 right and Q2 wrong.
        _row("A6", "Q1", ndcg_at_k=0.9),
        _row("A6", "Q2", ndcg_at_k=0.1),
    ]
    gap = oracle_gap(rows, "A6", ["A1", "A5"])
    assert gap is not None
    assert gap.oracle_score == pytest.approx(0.9)
    assert gap.router_score == pytest.approx(0.5)
    assert gap.gap == pytest.approx(0.4)


def test_a_perfect_router_has_no_gap() -> None:
    rows = [
        _row("A1", "Q1", ndcg_at_k=0.9),
        _row("A5", "Q1", ndcg_at_k=0.1),
        _row("A6", "Q1", ndcg_at_k=0.9),
    ]
    gap = oracle_gap(rows, "A6", ["A1", "A5"])
    assert gap is not None
    assert gap.gap == pytest.approx(0.0)


def test_reports_whether_the_router_beat_the_best_fixed_arm() -> None:
    """Beating fixed strategies is the weaker claim, but it still has to hold."""
    rows = [
        _row("A1", "Q1", ndcg_at_k=0.9),
        _row("A1", "Q2", ndcg_at_k=0.1),
        _row("A5", "Q1", ndcg_at_k=0.1),
        _row("A5", "Q2", ndcg_at_k=0.9),
        _row("A6", "Q1", ndcg_at_k=0.9),
        _row("A6", "Q2", ndcg_at_k=0.9),
    ]
    gap = oracle_gap(rows, "A6", ["A1", "A5"])
    assert gap is not None
    assert gap.best_fixed_score == pytest.approx(0.5)
    assert gap.beats_best_fixed is True


def test_oracle_gap_is_none_when_the_router_has_no_rows() -> None:
    assert oracle_gap([_row("A1", "Q1")], "A6", ["A1"]) is None
