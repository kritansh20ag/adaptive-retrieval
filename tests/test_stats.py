from __future__ import annotations

import math

import pytest

from adaptive_retrieval.stats import (
    bonferroni_adjust,
    noise_floor,
    paired_bootstrap,
    pass_at_k,
    pass_caret_k,
)

# --------------------------------------------------------------------------
# noise floor - the number computed before spending money on a run
# --------------------------------------------------------------------------


def test_noise_floor_matches_the_documented_figures() -> None:
    assert noise_floor(120, 1) == pytest.approx(0.0913, abs=1e-4)
    assert noise_floor(120, 2) == pytest.approx(0.0645, abs=1e-4)
    assert noise_floor(120, 3) == pytest.approx(0.0527, abs=1e-4)
    # One query class in isolation is much noisier than the whole set.
    assert noise_floor(30, 2) == pytest.approx(0.1291, abs=1e-4)


def test_noise_floor_shrinks_with_more_data() -> None:
    assert noise_floor(240, 2) < noise_floor(120, 2)
    assert noise_floor(120, 4) < noise_floor(120, 2)


@pytest.mark.parametrize(("cases", "reps"), [(0, 1), (1, 0), (-1, 1)])
def test_noise_floor_rejects_invalid_sizes(cases: int, reps: int) -> None:
    with pytest.raises(ValueError):
        noise_floor(cases, reps)


# --------------------------------------------------------------------------
# paired bootstrap
# --------------------------------------------------------------------------


def test_identical_arms_show_no_difference_and_no_significance() -> None:
    scores = [0.5, 0.7, 0.2, 0.9, 0.4] * 10
    result = paired_bootstrap(scores, scores, resamples=2000)
    assert result.mean_difference == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)
    assert not result.is_significant()


def test_constant_uniform_improvement_is_detected() -> None:
    b = [0.4] * 60
    a = [0.6] * 60
    result = paired_bootstrap(a, b, resamples=2000)
    assert result.mean_difference == pytest.approx(0.2)
    assert result.is_significant()
    assert result.ci_low > 0


def test_pure_noise_is_not_significant() -> None:
    import random

    rng = random.Random(7)
    a = [rng.random() for _ in range(120)]
    b = [rng.random() for _ in range(120)]
    result = paired_bootstrap(a, b, resamples=2000)
    assert not result.is_significant()


def test_interval_brackets_the_observed_difference() -> None:
    a = [0.9, 0.8, 0.7, 0.6, 0.5] * 12
    b = [0.5, 0.5, 0.5, 0.5, 0.5] * 12
    result = paired_bootstrap(a, b, resamples=2000)
    assert result.ci_low <= result.mean_difference <= result.ci_high


def test_p_value_is_never_exactly_zero() -> None:
    """Add-one smoothing: the data cannot support a claim of p == 0."""
    a = [1.0] * 100
    b = [0.0] * 100
    result = paired_bootstrap(a, b, resamples=1000)
    assert result.p_value > 0
    assert result.p_value == pytest.approx(1 / 1001)


def test_is_deterministic_for_a_given_seed() -> None:
    a = [0.1, 0.9, 0.5, 0.3, 0.7] * 20
    b = [0.2, 0.4, 0.6, 0.1, 0.8] * 20
    first = paired_bootstrap(a, b, resamples=1000, seed=42)
    second = paired_bootstrap(a, b, resamples=1000, seed=42)
    assert first == second


def test_different_seeds_give_similar_but_distinct_intervals() -> None:
    a = [0.1, 0.9, 0.5, 0.3, 0.7] * 20
    b = [0.2, 0.4, 0.6, 0.1, 0.8] * 20
    first = paired_bootstrap(a, b, resamples=1000, seed=1)
    second = paired_bootstrap(a, b, resamples=1000, seed=2)
    assert first.mean_difference == pytest.approx(second.mean_difference)
    assert first.ci_low == pytest.approx(second.ci_low, abs=0.05)


def test_direction_is_a_minus_b() -> None:
    worse = paired_bootstrap([0.1] * 20, [0.5] * 20, resamples=500)
    better = paired_bootstrap([0.5] * 20, [0.1] * 20, resamples=500)
    assert worse.mean_difference < 0 < better.mean_difference


def test_wider_confidence_gives_wider_interval() -> None:
    a = [0.9, 0.2, 0.7, 0.4, 0.6] * 12
    b = [0.5] * 60
    narrow = paired_bootstrap(a, b, resamples=2000, confidence=0.80)
    wide = paired_bootstrap(a, b, resamples=2000, confidence=0.99)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        paired_bootstrap([], [])


def test_n_pairs_is_reported() -> None:
    assert paired_bootstrap([0.1] * 17, [0.2] * 17, resamples=200).n_pairs == 17


# --------------------------------------------------------------------------
# multiple comparisons
# --------------------------------------------------------------------------


def test_bonferroni_scales_by_family_size() -> None:
    assert bonferroni_adjust([0.01, 0.02]) == pytest.approx([0.02, 0.04])


def test_bonferroni_clamps_at_one() -> None:
    assert bonferroni_adjust([0.5, 0.9]) == [1.0, 1.0]


def test_bonferroni_of_empty_family() -> None:
    assert bonferroni_adjust([]) == []


def test_bonferroni_can_flip_a_borderline_result() -> None:
    """With 36 pairwise comparisons across 9 arms, p=0.04 is not significant."""
    adjusted = bonferroni_adjust([0.04] * 36)
    assert all(p == 1.0 for p in adjusted)


# --------------------------------------------------------------------------
# non-determinism metrics
# --------------------------------------------------------------------------


def test_pass_at_k_needs_only_one_success() -> None:
    assert pass_at_k(n_trials=3, n_correct=1, k=3) == 1.0
    assert pass_at_k(n_trials=3, n_correct=0, k=1) == 0.0


def test_pass_caret_k_needs_every_success() -> None:
    assert pass_caret_k(n_trials=3, n_correct=3, k=3) == 1.0
    assert pass_caret_k(n_trials=3, n_correct=2, k=3) == 0.0


def test_pass_at_k_exceeds_pass_caret_k_when_inconsistent() -> None:
    at_k = pass_at_k(n_trials=4, n_correct=2, k=2)
    caret_k = pass_caret_k(n_trials=4, n_correct=2, k=2)
    assert at_k > caret_k


def test_pass_at_one_equals_pass_caret_one() -> None:
    for correct in range(5):
        assert pass_at_k(4, correct, 1) == pytest.approx(pass_caret_k(4, correct, 1))


def test_unbiased_estimator_differs_from_the_naive_form() -> None:
    """1 - C(n-c,k)/C(n,k) is not 1 - (1 - c/n)^k, and the difference matters
    at the rep counts a benchmark can afford."""
    n, c, k = 4, 2, 2
    naive = 1 - (1 - c / n) ** k
    assert pass_at_k(n, c, k) == pytest.approx(1 - math.comb(2, 2) / math.comb(4, 2))
    assert pass_at_k(n, c, k) != pytest.approx(naive)


@pytest.mark.parametrize(
    ("trials", "correct", "k"),
    [(0, 0, 1), (3, 4, 1), (3, -1, 1), (3, 2, 0), (3, 2, 4)],
)
def test_invalid_trial_counts_rejected(trials: int, correct: int, k: int) -> None:
    with pytest.raises(ValueError):
        pass_at_k(trials, correct, k)
