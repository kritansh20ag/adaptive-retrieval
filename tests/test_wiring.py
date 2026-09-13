from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from adaptive_retrieval.analysis import load_rows, summarise
from adaptive_retrieval.cli import main
from adaptive_retrieval.config import load_config
from adaptive_retrieval.golden import GoldenCase
from adaptive_retrieval.harness.wiring import NullArm, OracleArm, smoke_verdict
from adaptive_retrieval.judge import KeywordEntailment

CONFIG = """
corpus: test-corpus
golden_set: golden/v1.jsonl
judges:
  entailment: deberta-v3-large-mnli
defaults:
  index: chunks-v1
  k: 10
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
        "id": "Q1",
        "class": "multi_hop",
        "question": "Which outlets covered both?",
        "gold_chunks": ["g1", "g2"],
        "answer": "Reuters and the Financial Times.",
        "should_abstain": False,
    }
)

UNANSWERABLE = GoldenCase.model_validate(
    {
        "id": "Q2",
        "class": "unanswerable",
        "question": "What was the headcount?",
        "gold_chunks": [],
        "answer": None,
        "should_abstain": True,
    }
)


def _config(tmp_path: Path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(textwrap.dedent(CONFIG), encoding="utf-8")
    return load_config(path)


# --------------------------------------------------------------------------
# the oracle: proves the grading pipeline can score a perfect answer
# --------------------------------------------------------------------------


def test_oracle_returns_the_gold_chunks(tmp_path: Path) -> None:
    outcome = OracleArm(_config(tmp_path))("ORACLE", ANSWERABLE)
    assert outcome.retrieval.chunk_ids == ["g1", "g2"]


def test_oracle_abstains_on_unanswerable_questions(tmp_path: Path) -> None:
    outcome = OracleArm(_config(tmp_path))("ORACLE", UNANSWERABLE)
    assert outcome.generation.payload.abstained is True
    assert outcome.retrieval.chunks == ()


def test_oracle_cites_the_gold_chunks(tmp_path: Path) -> None:
    outcome = OracleArm(_config(tmp_path), entails=KeywordEntailment())("ORACLE", ANSWERABLE)
    assert outcome.citations.recall == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the null: proves the grader is not too lenient
# --------------------------------------------------------------------------


def test_null_retrieves_and_answers_nothing(tmp_path: Path) -> None:
    outcome = NullArm(_config(tmp_path))("NULL", ANSWERABLE)
    assert outcome.retrieval.chunks == ()
    assert outcome.generation.payload.abstained is True


def test_null_abstains_even_on_answerable_questions(tmp_path: Path) -> None:
    """A grader that scores this above chance on abstention is one-sided."""
    outcome = NullArm(_config(tmp_path))("NULL", ANSWERABLE)
    assert outcome.generation.payload.abstained is True


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------


class _Summary:
    def __init__(self, arm: str, ndcg: float | None) -> None:
        self.arm = arm
        self.ndcg = ndcg


def test_smoke_passes_when_oracle_is_perfect_and_null_is_zero() -> None:
    passed, complaints = smoke_verdict([_Summary("ORACLE", 1.0), _Summary("NULL", 0.0)])
    assert passed
    assert complaints == []


def test_smoke_fails_when_the_grader_cannot_score_a_perfect_answer() -> None:
    passed, complaints = smoke_verdict([_Summary("ORACLE", 0.6), _Summary("NULL", 0.0)])
    assert not passed
    assert "grader cannot score a perfect answer" in complaints[0]


def test_smoke_fails_when_the_grader_rewards_an_empty_answer() -> None:
    passed, complaints = smoke_verdict([_Summary("ORACLE", 1.0), _Summary("NULL", 0.4)])
    assert not passed
    assert "too lenient" in complaints[0]


def test_smoke_fails_when_an_arm_produced_nothing() -> None:
    passed, complaints = smoke_verdict([_Summary("ORACLE", 1.0)])
    assert not passed
    assert "no NULL rows" in complaints[0]


# --------------------------------------------------------------------------
# end to end through the CLI - no cluster, no model, no spend
# --------------------------------------------------------------------------


def test_smoke_command_runs_and_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--config", "config/benchmark.yaml", "smoke", "--out", str(tmp_path)])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "smoke OK" in out

    run_dir = next(tmp_path.iterdir())
    rows = load_rows(run_dir / "results.jsonl")
    summaries = {s.arm: s for s in summarise(rows)}

    assert summaries["ORACLE"].ndcg == pytest.approx(1.0)
    assert summaries["NULL"].ndcg == pytest.approx(0.0)
    # An always-abstain baseline can score no better than the base rate of
    # unanswerable questions in the set.
    assert summaries["NULL"].abstention_accuracy is not None
    assert summaries["NULL"].abstention_accuracy < 0.5


def test_run_without_yes_spends_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The estimate path must not touch a cluster or a model."""
    exit_code = main(["--config", "config/benchmark.yaml", "run", "--out", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Nothing has been spent" in out
    assert "noise floor" in out
    assert not list(tmp_path.iterdir())
