"""The result row, the error sidecar, and the invariants that separate them.

The harness's whole output is one row per ``(arm, query, rep)``. That row is
the project's actual contribution: quality frameworks emit relevance without
cost, observability tools emit cost without relevance judgements, and nobody
ships the join. So the schema is load-bearing, and the rules below are enforced
in code rather than left to the caller's discipline.

Three invariants
----------------
1. **An attempt that produced no scorable output is not a result.** It goes to
   ``errors.jsonl`` with a failure class. If it went to ``results.jsonl`` it
   would occupy the ``(arm, query, rep)`` slot, block resume, and score
   plumbing as a model failure. ``ResultRow`` therefore has no way to express
   "this failed" - that state lives in a different type, in a different file.

2. **"Abstained" is not "produced nothing".** ``abstained`` must come from an
   explicitly parsed refusal, never from an empty or unparseable answer. This
   matters more here than in most harnesses because a whole query class is
   unanswerable: if the two shared a label, a runner that errored on every
   input would score identically to one that correctly found nothing.

3. **Judge cost is not arm cost.** They are separate fields. Folding the
   judge's spend into the arm's would dampen exactly the differences between
   arms that the benchmark exists to measure.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ErrorRecord",
    "FailureClass",
    "ResultRow",
    "RowStatus",
    "RunWriter",
    "StageLatency",
    "TokenUsage",
]


class FailureClass(StrEnum):
    """Why an attempt produced no scorable output.

    Every one of these is a fact about the plumbing, not about the model.
    """

    HARNESS_ERROR = "harness_error"
    SERVING_ERROR = "serving_error"
    TIMEOUT = "timeout"
    MODEL_MISMATCH = "model_mismatch"
    GRADER_ERROR = "grader_error"
    UNPARSEABLE_OUTPUT = "unparseable_output"


class RowStatus(StrEnum):
    OK = "ok"
    #: The response hit ``max_tokens``. Counted and shown, but not averaged in
    #: as though the model chose to stop - truncation is easy to misread as a
    #: wrong answer.
    TRUNCATED = "truncated"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StageLatency(_Base):
    """Milliseconds per stage, plus total wall-clock for the case.

    Timed on the final successful request only. Retries, backoff sleeps and
    local queueing are excluded, or whichever arm hit more transient errors
    would look slower than it is.

    Note there is deliberately **no** "total >= sum of stages" rule. Stages can
    overlap: the three retrieval legs run concurrently inside a single
    Elasticsearch ``_search``, so their measured times legitimately sum to more
    than the wall-clock they occupied. The check that *is* always true under
    concurrency is that no single stage exceeds the total, and that is what
    catches a units error or a double-count.
    """

    route: float = Field(default=0.0, ge=0.0)
    retrieve: float = Field(default=0.0, ge=0.0)
    rerank: float = Field(default=0.0, ge=0.0)
    graph: float = Field(default=0.0, ge=0.0)
    generate: float = Field(default=0.0, ge=0.0)
    total: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _no_stage_exceeds_total(self) -> Self:
        """The one relation that survives concurrency.

        Stages may overlap, so their SUM is unbounded by the total - but no
        single stage can outlast the case that contains it. This is the
        check that catches a units error or a double-count.
        """
        for name in ("route", "retrieve", "rerank", "graph", "generate"):
            value = getattr(self, name)
            if value > self.total + 1e-6:
                raise ValueError(
                    f"stage {name}={value}ms exceeds total={self.total}ms - a stage is "
                    f"double-counted, or the units differ"
                )
        return self

    @property
    def stage_sum(self) -> float:
        """Sum of stages. May exceed ``total`` when stages ran concurrently."""
        return self.route + self.retrieve + self.rerank + self.graph + self.generate


class TokenUsage(_Base):
    """Token counts read from the API's ``usage`` block, never estimated.

    String-length estimates are off by enough to reverse a cost comparison.
    """

    input: int = Field(ge=0)
    output: int = Field(ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_write: int = Field(default=0, ge=0)


class ResultRow(_Base):
    """One scorable attempt. Written to ``results.jsonl`` and indexed."""

    run_id: str
    arm: str
    query_id: str
    query_class: str
    rep: int = Field(ge=0)

    #: The model that actually served the request, read off the response. A
    #: provider fallback or capacity reroute silently invalidates a run.
    served_model: str
    stop_reason: str | None = None
    status: RowStatus = RowStatus.OK

    # --- routing, so A6 is auditable rather than magic ---
    route_taken: str | None = None
    route_signal: dict[str, Any] | None = None
    retried: bool = False
    attempts: int = Field(default=1, ge=1)

    # --- retrieval quality. None means undefined, not zero. ---
    hit_rate_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)

    # --- trust ---
    citation_precision: float | None = None
    citation_recall: float | None = None
    judge_disagreement: bool | None = None
    #: Must come from a parsed structured refusal, never from empty output.
    abstained: bool
    should_abstain: bool

    # --- operations ---
    latency_ms: StageLatency
    tokens: TokenUsage
    #: ``None`` when the served model has no published price. Never 0.0 as a
    #: stand-in: a silent zero would make that arm look free.
    cost_usd: float | None = Field(default=None, ge=0.0)

    # --- the judge is metered separately from the arm ---
    judge_model: str | None = None
    judge_tokens: TokenUsage | None = None
    judge_cost_usd: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        # An abstention is a decision, so there is nothing retrieved to score
        # against and no answer to cite. Claiming both is incoherent.
        if self.abstained and self.citation_precision is not None:
            raise ValueError(
                f"{self.arm}/{self.query_id}: abstained rows cannot carry a "
                f"citation_precision - there is no answer to cite"
            )
        if self.retried and self.attempts < 2:
            raise ValueError(
                f"{self.arm}/{self.query_id}: retried=True but attempts={self.attempts}"
            )
        # A router arm must say where it routed, or A6 is unauditable.
        if self.route_signal is not None and self.route_taken is None:
            raise ValueError(
                f"{self.arm}/{self.query_id}: route_signal recorded without route_taken"
            )
        return self

    @property
    def total_cost_usd(self) -> float | None:
        """Arm cost plus judge cost. Reported separately; summed only here."""
        if self.cost_usd is None:
            return None
        return self.cost_usd + self.judge_cost_usd


class ErrorRecord(_Base):
    """An attempt that never produced a scorable output.

    Deliberately not a ``ResultRow``: there is no score field to fill in, so
    there is no way for one of these to be averaged into a headline by mistake.
    """

    run_id: str
    arm: str
    query_id: str
    rep: int = Field(ge=0)
    failure: FailureClass
    message: str
    attempts: int = Field(default=1, ge=1)
    #: Tokens are still recorded - a failed attempt costs real money and must
    #: appear in "attempts run vs attempts scored".
    tokens: TokenUsage | None = None
    cost_usd: float | None = Field(default=None, ge=0.0)


class RunWriter:
    """Append-only writer for one run's results and errors.

    Results and errors go to *different files* by construction. There is no
    code path that writes an error into the results stream.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._results_path = self.run_dir / "results.jsonl"
        self._errors_path = self.run_dir / "errors.jsonl"
        self._results = self._results_path.open("a", encoding="utf-8")
        self._errors = self._errors_path.open("a", encoding="utf-8")

    @property
    def results_path(self) -> Path:
        return self._results_path

    @property
    def errors_path(self) -> Path:
        return self._errors_path

    def write_result(self, row: ResultRow) -> None:
        self._results.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
        self._results.flush()

    def write_error(self, record: ErrorRecord) -> None:
        self._errors.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        self._errors.flush()

    def completed_keys(self) -> set[tuple[str, str, int]]:
        """``(arm, query_id, rep)`` triples already scored, for resume.

        Only results count. An attempt that errored is *not* complete and will
        be retried on resume - which is only correct because errors were never
        written into the results stream.
        """
        if not self._results_path.exists():
            return set()
        keys: set[tuple[str, str, int]] = set()
        with self._results_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                keys.add((record["arm"], record["query_id"], record["rep"]))
        return keys

    def close(self) -> None:
        self._results.close()
        self._errors.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
