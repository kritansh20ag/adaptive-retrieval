"""Turning result rows into the scorecard.

Two things here that most RAG comparisons do not report:

* **Lift over closed book.** Every arm's quality is reported relative to A0,
  not to zero. On a corpus of real entities the model can answer some questions
  from parametric memory, and without that control every arm's apparent
  contribution is inflated by an unknown amount.

* **The routing oracle gap.** For each question we know which arm actually
  scored best, so we can compute what a perfect router would have achieved.
  The distance between A6 and that oracle is the honest measure of how much
  routing headroom is left - published routers sit around 43.7% against an
  oracle's 60.8%, so a gap is expected and hiding it would be the anomaly.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from adaptive_retrieval.stats import BootstrapResult, bonferroni_adjust, paired_bootstrap

__all__ = [
    "ArmSummary",
    "OracleGap",
    "compare_arms",
    "load_rows",
    "oracle_gap",
    "summarise",
]


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``results.jsonl``. Errors live in a different file by design."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _defined(values: Iterable[Any]) -> list[float]:
    """Drop ``None`` rather than coercing it to 0.

    A None is an undefined metric - an unanswerable question has no retrieval
    score. Averaging it in as zero would drag every arm down by the same amount
    and compress the differences the benchmark exists to detect.
    """
    return [float(v) for v in values if v is not None]


@dataclass(frozen=True, slots=True)
class ArmSummary:
    arm: str
    n_rows: int
    n_scored: int
    ndcg: float | None
    hit_rate: float | None
    mrr: float | None
    citation_recall: float | None
    abstention_accuracy: float | None
    mean_cost_usd: float
    mean_judge_cost_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    retry_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "n_rows": self.n_rows,
            "n_scored": self.n_scored,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "citation_recall": self.citation_recall,
            "abstention_accuracy": self.abstention_accuracy,
            "mean_cost_usd": self.mean_cost_usd,
            "mean_judge_cost_usd": self.mean_judge_cost_usd,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "retry_rate": self.retry_rate,
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def summarise(
    rows: Sequence[dict[str, Any]], *, query_class: str | None = None
) -> list[ArmSummary]:
    """One summary per arm, optionally restricted to a single query class."""
    selected = [r for r in rows if query_class is None or r["query_class"] == query_class]
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_arm[row["arm"]].append(row)

    summaries: list[ArmSummary] = []
    for arm in sorted(by_arm):
        arm_rows = by_arm[arm]
        ndcg = _defined(r.get("ndcg_at_k") for r in arm_rows)
        hits = _defined(r.get("hit_rate_at_k") for r in arm_rows)
        mrrs = _defined(r.get("mrr") for r in arm_rows)
        recalls = _defined(r.get("citation_recall") for r in arm_rows)
        latencies = [float(r["latency_ms"]["total"]) for r in arm_rows]
        # Abstention is scored on every row, including the answerable ones -
        # a system that abstains when it should not is failing too, and a
        # one-sided metric would reward always-abstain.
        abstention = [
            1.0 if bool(r["abstained"]) == bool(r["should_abstain"]) else 0.0 for r in arm_rows
        ]

        summaries.append(
            ArmSummary(
                arm=arm,
                n_rows=len(arm_rows),
                n_scored=len(ndcg),
                ndcg=fmean(ndcg) if ndcg else None,
                hit_rate=fmean(hits) if hits else None,
                mrr=fmean(mrrs) if mrrs else None,
                citation_recall=fmean(recalls) if recalls else None,
                abstention_accuracy=fmean(abstention) if abstention else None,
                mean_cost_usd=fmean([float(r["cost_usd"]) for r in arm_rows]),
                mean_judge_cost_usd=fmean([float(r.get("judge_cost_usd", 0.0)) for r in arm_rows]),
                p50_latency_ms=_percentile(latencies, 0.50),
                p95_latency_ms=_percentile(latencies, 0.95),
                retry_rate=fmean([1.0 if r.get("retried") else 0.0 for r in arm_rows]),
            )
        )
    return summaries


def _paired_scores(
    rows: Sequence[dict[str, Any]], arm_a: str, arm_b: str, metric: str
) -> tuple[list[float], list[float]]:
    """Align two arms on ``(query_id, rep)``, keeping only pairs both scored."""
    index: dict[tuple[str, str, int], float] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        index[(row["arm"], row["query_id"], int(row["rep"]))] = float(value)

    keys = sorted(
        {(q, rep) for (arm, q, rep) in index if arm == arm_a}
        & {(q, rep) for (arm, q, rep) in index if arm == arm_b}
    )
    return (
        [index[(arm_a, q, rep)] for q, rep in keys],
        [index[(arm_b, q, rep)] for q, rep in keys],
    )


def compare_arms(
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[tuple[str, str]],
    *,
    metric: str = "ndcg_at_k",
    resamples: int = 10_000,
    seed: int = 20260905,
) -> list[tuple[tuple[str, str], BootstrapResult, float]]:
    """Paired-bootstrap every pair, Bonferroni-adjusted across the family.

    Returns ``((a, b), result, adjusted_p)``. Compare the *adjusted* p against
    the original alpha - with nine arms there are 36 pairwise comparisons and
    roughly two will look significant by chance.
    """
    results: list[tuple[tuple[str, str], BootstrapResult]] = []
    for arm_a, arm_b in pairs:
        scores_a, scores_b = _paired_scores(rows, arm_a, arm_b, metric)
        if not scores_a:
            continue
        results.append(
            ((arm_a, arm_b), paired_bootstrap(scores_a, scores_b, resamples=resamples, seed=seed))
        )

    adjusted = bonferroni_adjust([r.p_value for _, r in results])
    return [(pair, result, p) for (pair, result), p in zip(results, adjusted, strict=True)]


@dataclass(frozen=True, slots=True)
class OracleGap:
    """How far a router is from a perfect one."""

    router_arm: str
    router_score: float
    oracle_score: float
    best_fixed_arm: str
    best_fixed_score: float
    n_questions: int

    @property
    def gap(self) -> float:
        return self.oracle_score - self.router_score

    @property
    def beats_best_fixed(self) -> bool:
        return self.router_score > self.best_fixed_score


def oracle_gap(
    rows: Sequence[dict[str, Any]],
    router_arm: str,
    candidate_arms: Sequence[str],
    *,
    metric: str = "ndcg_at_k",
) -> OracleGap | None:
    """Compare a router against a perfect router over the same questions.

    The oracle picks, per question, whichever candidate arm actually scored
    best. It is unattainable by construction - that is the point. The gap is
    the honest statement of how much routing headroom remains.
    """
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            scores[(row["arm"], row["query_id"])].append(float(value))

    questions = sorted({q for (arm, q) in scores if arm == router_arm})
    if not questions:
        return None

    router_values: list[float] = []
    oracle_values: list[float] = []
    for question in questions:
        candidates = [
            fmean(scores[(arm, question)]) for arm in candidate_arms if (arm, question) in scores
        ]
        if not candidates:
            continue
        router_values.append(fmean(scores[(router_arm, question)]))
        oracle_values.append(max(candidates))

    if not router_values:
        return None

    fixed_means = {
        arm: fmean([v for (a, q), vs in scores.items() if a == arm for v in vs])
        for arm in candidate_arms
        if any(a == arm for (a, _) in scores)
    }
    best_fixed = max(fixed_means, key=lambda a: fixed_means[a])

    return OracleGap(
        router_arm=router_arm,
        router_score=fmean(router_values),
        oracle_score=fmean(oracle_values),
        best_fixed_arm=best_fixed,
        best_fixed_score=fixed_means[best_fixed],
        n_questions=len(router_values),
    )
