from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from adaptive_retrieval.config import load_config
from adaptive_retrieval.generate import AnswerPayload, GenerationError, GenerationResult
from adaptive_retrieval.golden import GoldenCase
from adaptive_retrieval.harness.row import RunWriter
from adaptive_retrieval.harness.runner import ArmOutcome, Runner, iter_trials
from adaptive_retrieval.metrics.citations import CitationScores
from adaptive_retrieval.retrieval.base import RetrievalResult, RetrievedChunk
from adaptive_retrieval.router.routers import RouteDecision

CONFIG = """
corpus: test-corpus
golden_set: golden/v1.jsonl
judges:
  entailment: deberta-v3-large-mnli
defaults:
  index: chunks-v1
  k: 10
run:
  reps: 2
arms:
  - id: A0
    kind: closed_book
  - id: A1
    kind: retrieval
    retriever:
      type: bm25
"""

ANSWERABLE = GoldenCase.model_validate(
    {
        "id": "Q17",
        "class": "multi_hop",
        "question": "Which outlets covered both?",
        "gold_chunks": ["c12", "c88"],
        "answer": "Reuters and the FT.",
        "should_abstain": False,
    }
)

UNANSWERABLE = GoldenCase.model_validate(
    {
        "id": "Q18",
        "class": "unanswerable",
        "question": "What was the headcount?",
        "gold_chunks": [],
        "answer": None,
        "should_abstain": True,
    }
)


def _config(tmp_path: Path, body: str = CONFIG):
    path = tmp_path / "benchmark.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_config(path)


def _outcome(
    *,
    chunk_ids: tuple[str, ...] = ("c05", "c12", "c40", "c88"),
    abstained: bool = False,
    route: RouteDecision | None = None,
    precision: float | None = 1.0,
) -> ArmOutcome:
    return ArmOutcome(
        # Stage times are sub-millisecond because the fake genuinely does no
        # work: StageLatency rejects a stage longer than the case that
        # contains it, and fabricating 200ms inside a 0.01ms call is exactly
        # the units error that validator exists to catch.
        retrieval=RetrievalResult(
            chunks=tuple(RetrievedChunk(cid, f"text {cid}", 1.0) for cid in chunk_ids),
            retrieve_ms=0.001,
        ),
        generation=GenerationResult(
            payload=AnswerPayload(abstained=abstained, sentences=[]),
            served_model="claude-opus-5",
            stop_reason="end_turn",
            input_tokens=4118,
            output_tokens=317,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=0.001,
            cost_usd=0.0141,
        ),
        citations=CitationScores(precision=precision, recall=1.0, n_statements=1, n_citations=1),
        route=route,
    )


def _rows(tmp_path: Path) -> list[dict]:
    return [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]


def _errors(tmp_path: Path) -> list[dict]:
    path = tmp_path / "errors.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


# --------------------------------------------------------------------------
# execution order
# --------------------------------------------------------------------------


def test_interleaved_order_is_question_major() -> None:
    """Arm-major order lands transient conditions entirely on one arm."""
    trials = list(iter_trials([ANSWERABLE, UNANSWERABLE], ["A0", "A1"], reps=1))
    assert [(c.id, a) for c, a, _ in trials] == [
        ("Q17", "A0"),
        ("Q17", "A1"),
        ("Q18", "A0"),
        ("Q18", "A1"),
    ]


def test_arm_major_order_is_available_but_not_the_default() -> None:
    trials = list(iter_trials([ANSWERABLE, UNANSWERABLE], ["A0", "A1"], 1, interleave=False))
    assert [(c.id, a) for c, a, _ in trials] == [
        ("Q17", "A0"),
        ("Q18", "A0"),
        ("Q17", "A1"),
        ("Q18", "A1"),
    ]


def test_every_rep_is_executed() -> None:
    trials = list(iter_trials([ANSWERABLE], ["A0"], reps=3))
    assert [rep for _, _, rep in trials] == [0, 1, 2]


def test_zero_reps_rejected() -> None:
    with pytest.raises(ValueError, match="reps must be >= 1"):
        list(iter_trials([ANSWERABLE], ["A0"], reps=0))


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def test_writes_one_row_per_arm_question_rep(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with RunWriter(tmp_path) as writer:
        rows, errors = Runner(config, writer, lambda a, c: _outcome(), run_id="r1").run(
            [ANSWERABLE]
        )
    assert (rows, errors) == (4, 0)  # 2 arms x 1 question x 2 reps


def test_metrics_are_computed_against_gold(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with RunWriter(tmp_path) as writer:
        Runner(config, writer, lambda a, c: _outcome(), run_id="r1").run([ANSWERABLE])
    row = _rows(tmp_path)[0]
    assert row["hit_rate_at_k"] == 1.0
    assert row["mrr"] == pytest.approx(0.5)
    assert row["ndcg_at_k"] == pytest.approx(0.6509, abs=1e-3)


def test_unanswerable_questions_get_null_retrieval_metrics(tmp_path: Path) -> None:
    """Undefined, not zero - scoring 0 would drag every arm down equally."""
    config = _config(tmp_path)
    with RunWriter(tmp_path) as writer:
        Runner(config, writer, lambda a, c: _outcome(abstained=True), run_id="r1").run(
            [UNANSWERABLE]
        )
    row = _rows(tmp_path)[0]
    assert row["ndcg_at_k"] is None
    assert row["mrr"] is None
    assert row["should_abstain"] is True
    assert row["abstained"] is True


def test_abstention_clears_citation_scores(tmp_path: Path) -> None:
    """There is no answer to cite."""
    config = _config(tmp_path)
    with RunWriter(tmp_path) as writer:
        Runner(config, writer, lambda a, c: _outcome(abstained=True), run_id="r1").run(
            [UNANSWERABLE]
        )
    assert _rows(tmp_path)[0]["citation_precision"] is None


def test_router_decisions_are_recorded(tmp_path: Path) -> None:
    config = _config(tmp_path)
    decision = RouteDecision(arm_id="A1", reason="dense", signal={"entities_found": 3})
    with RunWriter(tmp_path) as writer:
        Runner(config, writer, lambda a, c: _outcome(route=decision), run_id="r1").run([ANSWERABLE])
    row = _rows(tmp_path)[0]
    assert row["route_taken"] == "A1"
    assert row["route_signal"]["entities_found"] == 3


# --------------------------------------------------------------------------
# errors never become results
# --------------------------------------------------------------------------


def test_generation_error_goes_to_the_sidecar(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def failing(arm_id: str, case: GoldenCase) -> ArmOutcome:
        raise GenerationError("did not parse")

    with RunWriter(tmp_path) as writer:
        rows, errors = Runner(config, writer, failing, run_id="r1").run([ANSWERABLE])

    assert rows == 0
    assert errors == 4
    assert _errors(tmp_path)[0]["failure"] == "unparseable_output"
    assert not (tmp_path / "results.jsonl").read_text().strip()


def test_unexpected_exception_is_classified_as_harness_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def exploding(arm_id: str, case: GoldenCase) -> ArmOutcome:
        raise RuntimeError("connection reset")

    with RunWriter(tmp_path) as writer:
        Runner(config, writer, exploding, run_id="r1").run([ANSWERABLE])

    record = _errors(tmp_path)[0]
    assert record["failure"] == "harness_error"
    assert "connection reset" in record["message"]


def test_timeout_is_its_own_failure_class(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def timing_out(arm_id: str, case: GoldenCase) -> ArmOutcome:
        raise TimeoutError("stream hung")

    with RunWriter(tmp_path) as writer:
        Runner(config, writer, timing_out, run_id="r1").run([ANSWERABLE])

    assert _errors(tmp_path)[0]["failure"] == "timeout"


def test_one_failing_arm_does_not_stop_the_run(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def flaky(arm_id: str, case: GoldenCase) -> ArmOutcome:
        if arm_id == "A0":
            raise GenerationError("nope")
        return _outcome()

    with RunWriter(tmp_path) as writer:
        rows, errors = Runner(config, writer, flaky, run_id="r1").run([ANSWERABLE])

    assert rows == 2
    assert errors == 2


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def test_resume_skips_scored_trials(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with RunWriter(tmp_path) as writer:
        Runner(config, writer, lambda a, c: _outcome(), run_id="r1").run([ANSWERABLE])
    with RunWriter(tmp_path) as writer:
        rows, _ = Runner(config, writer, lambda a, c: _outcome(), run_id="r1").run([ANSWERABLE])
    assert rows == 0


def test_resume_retries_errored_trials(tmp_path: Path) -> None:
    """An errored attempt is not complete - which only works because errors
    never entered the results stream."""
    config = _config(tmp_path)

    def failing(arm_id: str, case: GoldenCase) -> ArmOutcome:
        raise GenerationError("transient")

    with RunWriter(tmp_path) as writer:
        Runner(config, writer, failing, run_id="r1").run([ANSWERABLE])
    with RunWriter(tmp_path) as writer:
        rows, _ = Runner(config, writer, lambda a, c: _outcome(), run_id="r1").run([ANSWERABLE])

    assert rows == 4


def test_resume_can_be_disabled(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with RunWriter(tmp_path) as writer:
        Runner(config, writer, lambda a, c: _outcome(), run_id="r1").run([ANSWERABLE])
    with RunWriter(tmp_path) as writer:
        rows, _ = Runner(config, writer, lambda a, c: _outcome(), run_id="r1").run(
            [ANSWERABLE], resume=False
        )
    assert rows == 4
