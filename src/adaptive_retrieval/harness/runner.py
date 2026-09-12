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

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from adaptive_retrieval.config import BenchmarkConfig, ClosedBookArm, RetrievalArm, RouterArm
from adaptive_retrieval.generate import GenerationError, GenerationResult
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
    judge_model: str | None = None
    judge_cost_usd: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_disagreement: bool | None = None


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
    ) -> None:
        self.config = config
        self.writer = writer
        self.run_arm = run_arm
        self.run_id = run_id

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
                outcome = self.run_arm(arm_id, case)
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
                self.writer.write_error(
                    ErrorRecord(
                        run_id=self.run_id,
                        arm=arm_id,
                        query_id=case.id,
                        rep=rep,
                        failure=FailureClass.HARNESS_ERROR,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
                errors += 1
                continue

            wall_ms = (time.perf_counter() - started) * 1000.0
            if wall_ms > self.config.run.max_case_seconds * 1000.0:
                # A hard wall-clock ceiling, independent of stream liveness: a
                # hung connection emitting keepalives defeats inactivity timers
                # forever. A timeout is an error, never a zero.
                self.writer.write_error(
                    ErrorRecord(
                        run_id=self.run_id,
                        arm=arm_id,
                        query_id=case.id,
                        rep=rep,
                        failure=FailureClass.TIMEOUT,
                        message=(
                            f"case exceeded {self.config.run.max_case_seconds}s ({wall_ms:.0f}ms)"
                        ),
                        tokens=TokenUsage(
                            input=outcome.generation.input_tokens,
                            output=outcome.generation.output_tokens,
                        ),
                        cost_usd=outcome.generation.cost_usd,
                    )
                )
                errors += 1
                continue

            self.writer.write_result(self._build_row(case, arm_id, rep, outcome, wall_ms))
            rows += 1

        return rows, errors

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
