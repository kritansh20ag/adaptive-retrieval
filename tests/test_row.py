from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_retrieval.harness.row import (
    ErrorRecord,
    FailureClass,
    ResultRow,
    RowStatus,
    RunWriter,
    StageLatency,
    TokenUsage,
)


def _row(**overrides: object) -> ResultRow:
    base: dict[str, object] = {
        "run_id": "r1",
        "arm": "A4",
        "query_id": "Q17",
        "query_class": "multi_hop",
        "rep": 0,
        "served_model": "claude-opus-5",
        "abstained": False,
        "should_abstain": False,
        "latency_ms": StageLatency(retrieve=214, rerank=583, generate=1902, total=2699),
        "tokens": TokenUsage(input=4118, output=317),
        "cost_usd": 0.0141,
    }
    base.update(overrides)
    return ResultRow(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# invariant 1: an error is not a result
# --------------------------------------------------------------------------


def test_result_row_has_no_way_to_express_failure() -> None:
    """Failure lives in a different type so it cannot be averaged into a
    headline by mistake."""
    assert "failure" not in ResultRow.model_fields
    assert "error" not in ResultRow.model_fields


def test_errors_and_results_go_to_different_files(tmp_path: Path) -> None:
    with RunWriter(tmp_path) as writer:
        writer.write_result(_row())
        writer.write_error(
            ErrorRecord(
                run_id="r1",
                arm="A4",
                query_id="Q18",
                rep=0,
                failure=FailureClass.TIMEOUT,
                message="exceeded 180s wall clock",
            )
        )

    results = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    errors = [json.loads(x) for x in (tmp_path / "errors.jsonl").read_text().splitlines()]
    assert [r["query_id"] for r in results] == ["Q17"]
    assert [e["query_id"] for e in errors] == ["Q18"]
    assert all("failure" not in r for r in results)


def test_errored_attempt_is_not_complete_and_will_be_retried(tmp_path: Path) -> None:
    """Resume must re-run an errored attempt, which only works because errors
    never entered the results stream."""
    with RunWriter(tmp_path) as writer:
        writer.write_result(_row(query_id="Q17"))
        writer.write_error(
            ErrorRecord(
                run_id="r1",
                arm="A4",
                query_id="Q18",
                rep=0,
                failure=FailureClass.SERVING_ERROR,
                message="429 after retries",
            )
        )

    with RunWriter(tmp_path) as writer:
        assert writer.completed_keys() == {("A4", "Q17", 0)}


def test_failed_attempts_still_record_their_cost() -> None:
    """A failed attempt costs real money and must show in attempts-run vs
    attempts-scored."""
    record = ErrorRecord(
        run_id="r1",
        arm="A4",
        query_id="Q18",
        rep=0,
        failure=FailureClass.UNPARSEABLE_OUTPUT,
        message="no JSON in response",
        tokens=TokenUsage(input=3000, output=12),
        cost_usd=0.0153,
    )
    assert record.cost_usd > 0


# --------------------------------------------------------------------------
# invariant 2: abstained is not "produced nothing"
# --------------------------------------------------------------------------


def test_abstention_and_citations_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="no answer to cite"):
        _row(abstained=True, citation_precision=0.5)


def test_abstention_is_recorded_alongside_the_ground_truth() -> None:
    """Both booleans are present, so a confusion matrix is computable."""
    correct = _row(abstained=True, should_abstain=True)
    hallucinated = _row(abstained=False, should_abstain=True)
    assert correct.abstained and correct.should_abstain
    assert not hallucinated.abstained and hallucinated.should_abstain


def test_truncation_is_its_own_status_not_a_wrong_answer() -> None:
    row = _row(status=RowStatus.TRUNCATED, stop_reason="max_tokens")
    assert row.status is RowStatus.TRUNCATED
    assert row.abstained is False


# --------------------------------------------------------------------------
# invariant 3: judge cost is not arm cost
# --------------------------------------------------------------------------


def test_judge_cost_is_a_separate_field() -> None:
    row = _row(judge_model="claude-opus-5", judge_cost_usd=0.004)
    assert row.cost_usd == pytest.approx(0.0141)
    assert row.judge_cost_usd == pytest.approx(0.004)
    assert row.total_cost_usd == pytest.approx(0.0181)


def test_judge_cost_defaults_to_zero_not_none() -> None:
    assert _row().judge_cost_usd == 0.0


# --------------------------------------------------------------------------
# metrics: None is undefined, not zero
# --------------------------------------------------------------------------


def test_undefined_retrieval_metrics_are_allowed() -> None:
    row = _row(should_abstain=True, abstained=True, ndcg_at_k=None, mrr=None)
    assert row.ndcg_at_k is None
    assert row.mrr is None


def test_zero_and_none_serialise_differently(tmp_path: Path) -> None:
    with RunWriter(tmp_path) as writer:
        writer.write_result(_row(query_id="Qa", ndcg_at_k=0.0))
        writer.write_result(_row(query_id="Qb", ndcg_at_k=None))
    rows = {
        json.loads(x)["query_id"]: json.loads(x)
        for x in (tmp_path / "results.jsonl").read_text().splitlines()
    }
    assert rows["Qa"]["ndcg_at_k"] == 0.0
    assert rows["Qb"]["ndcg_at_k"] is None


# --------------------------------------------------------------------------
# other guards
# --------------------------------------------------------------------------


def test_retried_requires_more_than_one_attempt() -> None:
    with pytest.raises(ValidationError, match="retried=True but attempts=1"):
        _row(retried=True, attempts=1)


def test_route_signal_requires_a_route() -> None:
    """A6 must be auditable: recording why without recording what is useless."""
    with pytest.raises(ValidationError, match="route_signal recorded without route_taken"):
        _row(route_signal={"entities_found": 3})


def test_router_row_records_both_decision_and_reason() -> None:
    row = _row(
        arm="A6",
        route_taken="A5",
        route_signal={"entities_found": 3, "avg_degree": 4.2, "in_lcc": True},
    )
    assert row.route_taken == "A5"
    assert row.route_signal is not None
    assert row.route_signal["in_lcc"] is True


def test_latency_total_must_cover_its_stages() -> None:
    with pytest.raises(ValidationError, match="less than the sum of stages"):
        StageLatency(retrieve=1000, generate=1000, total=500)


def test_latency_total_may_exceed_stages() -> None:
    """Unattributed overhead is fine; a total that is too small is a bug."""
    assert StageLatency(retrieve=100, generate=100, total=250).total == 250


def test_served_model_is_required() -> None:
    """A score served by the wrong model measures nothing."""
    with pytest.raises(ValidationError):
        ResultRow(  # type: ignore[call-arg]
            run_id="r1",
            arm="A4",
            query_id="Q17",
            query_class="multi_hop",
            rep=0,
            abstained=False,
            should_abstain=False,
            latency_ms=StageLatency(total=1.0),
            tokens=TokenUsage(input=1, output=1),
            cost_usd=0.0,
        )


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(nkdg_at_k=0.5)


def test_negative_cost_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(cost_usd=-0.01)


def test_rows_are_immutable() -> None:
    row = _row()
    with pytest.raises(ValidationError):
        row.cost_usd = 99.0  # type: ignore[misc]


def test_writer_appends_across_sessions(tmp_path: Path) -> None:
    with RunWriter(tmp_path) as writer:
        writer.write_result(_row(query_id="Q1"))
    with RunWriter(tmp_path) as writer:
        writer.write_result(_row(query_id="Q2"))
        assert writer.completed_keys() == {("A4", "Q1", 0), ("A4", "Q2", 0)}


def test_writer_creates_missing_directories(tmp_path: Path) -> None:
    nested = tmp_path / "runs" / "2026-09-08"
    with RunWriter(nested) as writer:
        writer.write_result(_row())
    assert (nested / "results.jsonl").exists()


def test_completed_keys_on_empty_run(tmp_path: Path) -> None:
    with RunWriter(tmp_path) as writer:
        assert writer.completed_keys() == set()
