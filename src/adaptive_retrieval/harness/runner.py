"""The run loop.

Nested for-loops and a resume file. Nothing about the control flow is decided
by a model, because every property the eval needs is incompatible with an
agentic runner: isolated state per trial, a deterministic cursor to resume
from, saved trajectories, and a clean separation between infra failure and
model failure.

Three invariants are enforced here rather than left to callers:

1. **Interleaved execution.** Question-major, not arm-major: every arm sees
   question 1, then every arm sees question 2. Running an arm to completion
   means each arm meets a different API load and cache state, and that
   difference lands entirely on whichever arm ran first.

2. **An attempt that produced no scorable output is not a result.** It goes to
   ``errors.jsonl`` with a failure class. It never occupies the
   ``(arm, query, rep)`` slot, so resume re-runs it rather than treating the
   plumbing failure as a model failure.

3. **Retrieval metrics are undefined, not zero, on unanswerable questions.**
   The ``None`` propagates from the metric functions to the row untouched.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from adaptive_retrieval.config import BenchmarkConfig, ClosedBookArm, RetrievalArm, RouterArm
from adaptive_retrieval.generate import GenerationError, GenerationResult, ModelMismatchError
from adaptive_retrieval.golden import GoldenCase
from adaptive_retrieval.harness.row import (
    ErrorRecord,
    FailureClass,
    ResultRow,
    RowStatus,
    RunWriter,
    StageLatency,
    TokenUsage,
)
from adaptive_retrieval.metrics.citations import CitationScores
from adaptive_retrieval.metrics.retrieval import hit_rate_at_k, ndcg_at_k, reciprocal_rank
from adaptive_retrieval.retrieval.base import RetrievalResult
from adaptive_retrieval.router.routers import RouteDecision

__all__ = ["ArmOutcome", "ArmRunner", "Runner", "iter_trials"]


#: Exception type names that mean the provider or the transport failed,
#: not this code. Matched by name so neither anthropic nor elastic_transport
#: becomes an import-time dependency of the runner.
_SERVING_ERRORS = frozenset(
    {
        "APIStatusError",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
        "ConnectionError",
        "ConnectionTimeout",
        "TransportError",
        "ApiError",
    }
)


def _classify(exc: BaseException) -> FailureClass:
    names = {type(exc).__name__} | {base.__name__ for base in type(exc).__mro__}
    if names & _SERVING_ERRORS:
        return FailureClass.SERVING_ERROR
    return FailureClass.HARNESS_ERROR


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """Everything one arm produced for one question."""

    retrieval: RetrievalResult
    generation: GenerationResult
    citations: CitationScores
    route: RouteDecision | None = None
    retried: bool = False
    attempts: int = 1
    route_ms: float = 0.0
    #: Latency of the discarded first attempt, kept out of the retrieval
    #: column so 'final successful request only' holds.
    retry_overhead_ms: float = 0.0
    judge_model: str | None = None
    judge_cost_usd: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_disagreement: bool | None = None
    #: Atomic answer-quality verdicts. None means the judge said unknown,
    #: which is excluded from aggregates rather than counted as a failure.
    faithful: bool | None = None
    relevant: bool | None = None


#: Runs one arm against one question. Injected so the loop is testable with no
#: cluster and no model.
ArmRunner = Callable[[str, GoldenCase], ArmOutcome]


def iter_trials(
    cases: Sequence[GoldenCase],
    arm_ids: Sequence[str],
    reps: int,
    *,
    interleave: bool = True,
) -> Iterator[tuple[GoldenCase, str, int]]:
    """Yield ``(case, arm_id, rep)`` in execution order.

    Interleaved order is question-major so transient conditions - rate limits,
    a warm cache, a slow afternoon - spread across every arm instead of
    landing on one.
    """
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    if interleave:
        for case in cases:
            for rep in range(reps):
                for arm_id in arm_ids:
                    yield case, arm_id, rep
    else:
        for arm_id in arm_ids:
            for case in cases:
                for rep in range(reps):
                    yield case, arm_id, rep


class Runner:
    """Drives the benchmark and writes one row per scorable attempt."""

    def __init__(
        self,
        config: BenchmarkConfig,
        writer: RunWriter,
        run_arm: ArmRunner,
        *,
        run_id: str,
        save_trajectories: bool = True,
    ) -> None:
        self.config = config
        self.writer = writer
        self.run_arm = run_arm
        self.run_id = run_id
        self.save_trajectories = save_trajectories
        # One worker: the deadline is enforced by not waiting past it, and a
        # pool lets a hung call be abandoned instead of blocking the run
        # forever. Checking elapsed time AFTER the call returns cannot
        # interrupt anything - it only catches slow-but-finished work.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ar-trial")

    def arm_ids(self) -> list[str]:
        return [arm.id for arm in self.config.arms]

    def run(self, cases: Sequence[GoldenCase], *, resume: bool = True) -> tuple[int, int]:
        """Execute every trial. Returns ``(rows_written, errors_written)``.

        Resume skips only *scored* trials. An attempt that errored is not
        complete and is retried - which is only correct because errors were
        never written into the results stream.
        """
        completed = self.writer.completed_keys() if resume else set()
        rows = errors = 0

        for case, arm_id, rep in iter_trials(
            cases, self.arm_ids(), self.config.run.reps, interleave=self.config.run.interleave
        ):
            if (arm_id, case.id, rep) in completed:
                continue

            started = time.perf_counter()
            try:
                future = self._pool.submit(self.run_arm, arm_id, case)
                try:
                    outcome = future.result(timeout=self.config.run.max_case_seconds)
                except FutureTimeout as exc:
                    # The worker thread may still be blocked; it is abandoned
                    # rather than joined, so one hung call cannot stop the run.
                    future.cancel()
                    raise TimeoutError(
                        f"case exceeded {self.config.run.max_case_seconds}s and was abandoned"
                    ) from exc
            except ModelMismatchError as exc:
                # Its own class: a substituted model invalidates the run, and
                # burying it in harness_error hides that from whoever reads
                # errors.jsonl looking for why the numbers moved.
                self.writer.write_error(
                    ErrorRecord(
                        run_id=self.run_id,
                        arm=arm_id,
                        query_id=case.id,
                        rep=rep,
                        failure=FailureClass.MODEL_MISMATCH,
                        message=str(exc),
                    )
                )
                errors += 1
                continue
            except GenerationError as exc:
                self.writer.write_error(
                    ErrorRecord(
                        run_id=self.run_id,
                        arm=arm_id,
                        query_id=case.id,
                        rep=rep,
                        failure=FailureClass.UNPARSEABLE_OUTPUT,
                        message=str(exc),
                    )
                )
                errors += 1
                continue
            except TimeoutError as exc:
                self.writer.write_error(
                    ErrorRecord(
                        run_id=self.run_id,
                        arm=arm_id,
                        query_id=case.id,
                        rep=rep,
                        failure=FailureClass.TIMEOUT,
                        message=str(exc),
                    )
                )
                errors += 1
                continue
            except Exception as exc:  # any failure here is plumbing, not the model
                # A provider outage and a bug in this repo are both "not the
                # model", but they are not the same thing to whoever reads
                # errors.jsonl asking why the numbers moved.
                self.writer.write_error(
                    ErrorRecord(
                        run_id=self.run_id,
                        arm=arm_id,
                        query_id=case.id,
                        rep=rep,
                        failure=_classify(exc),
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
                errors += 1
                continue

            wall_ms = (time.perf_counter() - started) * 1000.0
            row = self._build_row(case, arm_id, rep, outcome, wall_ms)
            self.writer.write_result(row)
            if self.save_trajectories:
                self._write_trajectory(row, case, outcome)
            rows += 1

        return rows, errors

    def close(self) -> None:
        """Release the worker pool. A hung trial's thread is abandoned."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _write_trajectory(self, row: ResultRow, case: GoldenCase, outcome: ArmOutcome) -> None:
        """Save everything needed to explain a score without re-running it.

        The highest-leverage habit for a debuggable eval: a surprising number
        can be traced to a fact about the model, or to a bug in the harness,
        from the saved artefacts alone.
        """
        directory = self.writer.run_dir / "trajectories" / row.arm
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": row.run_id,
            "arm": row.arm,
            "query_id": row.query_id,
            "rep": row.rep,
            "question": case.question,
            "gold_chunks": list(case.gold_chunks),
            "route": (
                {"arm_id": outcome.route.arm_id, "reason": outcome.route.reason}
                if outcome.route
                else None
            ),
            "retrieved": [
                {"chunk_id": c.chunk_id, "score": c.score, "text": c.text}
                for c in outcome.retrieval.chunks
            ],
            "graph_chunk_ids": list(outcome.retrieval.graph_chunk_ids),
            "answer": [
                {"text": s.text, "cited_chunk_ids": list(s.cited_chunk_ids)}
                for s in outcome.generation.payload.sentences
            ],
            "abstained": outcome.generation.payload.abstained,
            "stop_reason": outcome.generation.stop_reason,
            "citations": {
                "precision": outcome.citations.precision,
                "recall": outcome.citations.recall,
            },
        }
        (directory / f"{row.query_id}-rep{row.rep}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _build_row(
        self,
        case: GoldenCase,
        arm_id: str,
        rep: int,
        outcome: ArmOutcome,
        wall_ms: float,
    ) -> ResultRow:
        gold = list(case.gold_chunks)
        retrieved = outcome.retrieval.chunk_ids
        k = self.config.defaults.k

        # These return None when the question has no gold chunks. That None is
        # carried through untouched: scoring an unanswerable question as 0
        # would drag every arm's average down by the same amount and compress
        # the differences the benchmark exists to measure.
        hit = hit_rate_at_k(retrieved, gold, k)
        rr = reciprocal_rank(retrieved, gold, k)
        ndcg = ndcg_at_k(retrieved, gold, k)

        abstained = outcome.generation.payload.abstained
        return ResultRow(
            run_id=self.run_id,
            arm=arm_id,
            query_id=case.id,
            query_class=case.query_class.value,
            rep=rep,
            served_model=outcome.generation.served_model,
            stop_reason=outcome.generation.stop_reason,
            status=RowStatus.TRUNCATED if outcome.generation.truncated else RowStatus.OK,
            route_taken=outcome.route.arm_id if outcome.route else None,
            route_signal=dict(outcome.route.signal) if outcome.route else None,
            retried=outcome.retried,
            attempts=outcome.attempts,
            hit_rate_at_k=hit,
            mrr=rr,
            ndcg_at_k=ndcg,
            retrieved_chunk_ids=retrieved,
            # An abstention has no answer to cite, so citation scores are
            # undefined rather than zero.
            citation_precision=None if abstained else outcome.citations.precision,
            citation_recall=None if abstained else outcome.citations.recall,
            judge_disagreement=outcome.judge_disagreement,
            faithful=outcome.faithful,
            relevant=outcome.relevant,
            abstained=abstained,
            should_abstain=case.should_abstain,
            latency_ms=StageLatency(
                route=outcome.route_ms,
                retrieve=outcome.retrieval.retrieve_ms,
                rerank=outcome.retrieval.rerank_ms,
                graph=outcome.retrieval.graph_ms,
                generate=outcome.generation.latency_ms,
                total=wall_ms,
            ),
            tokens=TokenUsage(
                input=outcome.generation.input_tokens,
                output=outcome.generation.output_tokens,
                cache_read=outcome.generation.cache_read_tokens,
                cache_write=outcome.generation.cache_write_tokens,
            ),
            cost_usd=outcome.generation.cost_usd,
            judge_model=outcome.judge_model,
            judge_tokens=(
                TokenUsage(input=outcome.judge_input_tokens, output=outcome.judge_output_tokens)
                if outcome.judge_model
                else None
            ),
            # Recorded separately from the arm's cost: folding the judge's
            # spend into the arm's would dampen the differences between arms.
            judge_cost_usd=outcome.judge_cost_usd,
        )


def arm_kind(config: BenchmarkConfig, arm_id: str) -> str:
    """``closed_book`` | ``retrieval`` | ``router`` for one arm."""
    arm = config.arm(arm_id)
    if isinstance(arm, ClosedBookArm):
        return "closed_book"
    if isinstance(arm, RetrievalArm):
        return "retrieval"
    if isinstance(arm, RouterArm):
        return "router"
    raise TypeError(f"unknown arm type for {arm_id!r}")
