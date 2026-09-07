"""Significance testing for arm comparisons.

Every arm answers the *same* questions, so comparisons are **paired**: we work
with the per-question differences rather than with two independent averages.
Paired tests are far more sensitive, and they are the standard in IR
evaluation.

Three things this module exists to prevent:

1. **Reporting a difference smaller than the noise.** ``noise_floor`` is
   computed *before* the first paid run and compared against the effect we
   hope to detect. If the floor is larger, the answer is more reps, not more
   features.
2. **Reporting a bare p-value.** Every test here returns an effect size with a
   confidence interval. At n≈120 a p-value alone is fragile; "A6 beats A7 by
   0.04 nDCG, 95% CI [0.01, 0.07]" is not.
3. **Multiple-comparison inflation.** With 9 arms there are 36 pairwise
   comparisons, and at alpha 0.05 roughly two will look significant by chance
   alone. ``bonferroni_adjust`` is not optional.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "BootstrapResult",
    "bonferroni_adjust",
    "noise_floor",
    "paired_bootstrap",
    "pass_at_k",
    "pass_caret_k",
]


def noise_floor(n_cases: int, n_reps: int = 1) -> float:
    """Approximate half-width of the paired-difference 95% CI for a pass rate.

    ``1 / sqrt(n * R)``. This is a planning heuristic, not a measurement - it
    is for sizing the run before it happens. Once a baseline exists, use the
    interval from ``paired_bootstrap`` instead, which uses the observed
    variance rather than a worst case.

    120 cases x 2 reps gives about +/- 0.065, i.e. 6.5 points.
    """
    if n_cases < 1:
        raise ValueError(f"n_cases must be >= 1, got {n_cases}")
    if n_reps < 1:
        raise ValueError(f"n_reps must be >= 1, got {n_reps}")
    return 1.0 / math.sqrt(n_cases * n_reps)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Outcome of a paired bootstrap comparison of two arms."""

    #: Mean of (a - b) over the paired cases. The effect size, in metric units.
    mean_difference: float
    ci_low: float
    ci_high: float
    #: Two-sided bootstrap p-value against the null "no difference".
    p_value: float
    n_pairs: int

    @property
    def significant_at(self) -> float:
        return self.p_value

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def paired_bootstrap(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260905,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Compare two arms on the same questions by bootstrapping their differences.

    ``a_scores[i]`` and ``b_scores[i]`` must be the two arms' scores on the
    *same* question, in the same order. Pairing is the entire point; passing
    unaligned sequences produces a meaningless answer, so the lengths are
    checked but the alignment cannot be.

    The confidence interval comes from the percentiles of the resampled mean
    differences. The p-value is computed separately, against a null built by
    **centring** the differences on zero - resampling the uncentred
    differences and counting sign flips would answer a different question and
    is the classic way to get a bootstrap p-value wrong.
    """
    if len(a_scores) != len(b_scores):
        raise ValueError(
            f"paired comparison needs equal-length inputs, got {len(a_scores)} and {len(b_scores)}"
        )
    if not a_scores:
        raise ValueError("paired comparison needs at least one pair")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    differences = [a - b for a, b in zip(a_scores, b_scores, strict=True)]
    n = len(differences)
    observed = sum(differences) / n

    rng = random.Random(seed)

    # Interval: resample the observed differences.
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += differences[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    low_index = min(int(tail * resamples), resamples - 1)
    high_index = min(int((1.0 - tail) * resamples), resamples - 1)

    # p-value: resample from differences centred on zero, which is the null.
    centred = [d - observed for d in differences]
    rng_null = random.Random(seed + 1)
    at_least_as_extreme = 0
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += centred[rng_null.randrange(n)]
        if abs(total / n) >= abs(observed):
            at_least_as_extreme += 1

    # Add-one smoothing: a p-value of exactly 0 is not a claim the data can
    # support, and it misleads when resamples is small.
    p_value = (at_least_as_extreme + 1) / (resamples + 1)

    return BootstrapResult(
        mean_difference=observed,
        ci_low=means[low_index],
        ci_high=means[high_index],
        p_value=p_value,
        n_pairs=n,
    )


def bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    """Bonferroni-adjust a family of p-values, clamped to 1.0.

    With 9 arms there are 36 pairwise comparisons; at alpha 0.05 about two will
    look significant by chance. Compare the *adjusted* values against the
    original alpha.
    """
    if not p_values:
        return []
    count = len(p_values)
    return [min(1.0, p * count) for p in p_values]


def pass_at_k(n_trials: int, n_correct: int, k: int) -> float:
    """Probability that at least one of ``k`` sampled trials succeeds.

    Unbiased combinatorial estimator ``1 - C(n-c, k) / C(n, k)``, not the
    naive ``1 - (1 - c/n)^k``: with the handful of reps a benchmark can afford,
    the naive form is noticeably biased.

    Use this where one success is enough.
    """
    _validate_trials(n_trials, n_correct, k)
    failures = n_trials - n_correct
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(n_trials, k)


def pass_caret_k(n_trials: int, n_correct: int, k: int) -> float:
    """Probability that **all** ``k`` sampled trials succeed.

    Unbiased estimator ``C(c, k) / C(n, k)``. Use this where consistency
    matters - at a 75% per-trial success rate, all three trials pass only about
    42% of the time, and that gap is worth reporting rather than hiding behind
    a mean.
    """
    _validate_trials(n_trials, n_correct, k)
    if n_correct < k:
        return 0.0
    return math.comb(n_correct, k) / math.comb(n_trials, k)


def _validate_trials(n_trials: int, n_correct: int, k: int) -> None:
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if not 0 <= n_correct <= n_trials:
        raise ValueError(f"n_correct must be in [0, {n_trials}], got {n_correct}")
    if not 1 <= k <= n_trials:
        raise ValueError(f"k must be in [1, {n_trials}], got {k}")
