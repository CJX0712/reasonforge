"""
L0 invariants — numerical primitives.

Every test below asserts a mathematical property that must hold for *any*
correct implementation, so they double as a cross-check against a rewrite.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import math
import statistics

import pytest

from reasonforge.core.numeric import (
    DeterministicRNG,
    beta_sample,
    bootstrap_mean_ci,
    entropy,
    expected_calibration_error,
    gini_simpson,
    js_divergence,
    kl_divergence,
    logsumexp,
    normalized_entropy,
    softmax,
    top_k_filter,
    top_p_filter,
    ucb_score,
    wilson_interval,
)


# ---------------------------------------------------------------- logsumexp
def test_logsumexp_singleton():
    assert logsumexp([3.0]) == pytest.approx(3.0)


def test_logsumexp_shift_equivariance():
    xs = [1.0, 2.0, 3.0, -5.0]
    c = 17.0
    assert logsumexp([x + c for x in xs]) == pytest.approx(logsumexp(xs) + c, abs=1e-12)


def test_logsumexp_matches_naive():
    xs = [0.1, 0.2, 0.3]
    assert logsumexp(xs) == pytest.approx(math.log(sum(math.exp(x) for x in xs)), abs=1e-12)


def test_logsumexp_overflow_safe():
    # naive sum of exp would overflow; ours must not
    val = logsumexp([1000.0, 1001.0])
    assert math.isfinite(val)
    assert val == pytest.approx(1001.0 + math.log1p(math.exp(-1.0)), abs=1e-9)


def test_logsumexp_empty():
    assert logsumexp([]) == -math.inf


# ------------------------------------------------------------------ softmax
def test_softmax_is_distribution():
    p = softmax([1.0, 2.0, 3.0, 4.0])
    assert sum(p) == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 <= x <= 1.0 for x in p)


def test_softmax_monotone_in_logits():
    p = softmax([1.0, 5.0, 3.0])
    assert p[1] > p[2] > p[0]


def test_softmax_temperature_limits():
    hot = softmax([1.0, 2.0], temperature=10.0)
    cold = softmax([1.0, 2.0], temperature=0.01)
    assert abs(hot[0] - hot[1]) < abs(cold[0] - cold[1])
    assert cold[1] == pytest.approx(1.0, abs=1e-6)


def test_softmax_rejects_bad_temperature():
    with pytest.raises(ValueError):
        softmax([1.0], temperature=0.0)


# ------------------------------------------------------------------ entropy
def test_entropy_bounds():
    n = 8
    uniform = [1.0 / n] * n
    assert entropy(uniform, base=2.0) == pytest.approx(math.log2(n), abs=1e-12)
    assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-12)


def test_entropy_max_at_uniform():
    best = 0.0
    best_p = None
    for i in range(1, 10):
        p = [i / 10.0, 1 - i / 10.0]
        h = entropy(p)
        if h > best:
            best, best_p = h, p
    assert best_p[0] == pytest.approx(best_p[1], abs=0.05)


def test_normalized_entropy_range():
    assert 0.0 <= normalized_entropy([0.5, 0.5]) <= 1.0
    assert normalized_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert normalized_entropy([0.25] * 4) == pytest.approx(1.0)
    assert normalized_entropy([1.0]) == 0.0  # undefined -> 0


def test_gini_simpson_bounds():
    assert gini_simpson([1.0, 0.0]) == pytest.approx(0.0)
    assert gini_simpson([0.5, 0.5]) == pytest.approx(0.5)
    n = 4
    assert gini_simpson([1.0 / n] * n) == pytest.approx(1 - 1 / n)


# --------------------------------------------------------------- divergence
def test_kl_nonnegative_and_zero_on_identical():
    p = [0.4, 0.3, 0.3]
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert kl_divergence(p, [0.3, 0.3, 0.4]) >= 0.0


def test_kl_asymmetric():
    p, q = [0.9, 0.1], [0.5, 0.5]
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


def test_js_symmetric_and_bounded():
    p, q = [0.8, 0.2], [0.3, 0.7]
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-12)
    assert 0.0 <= js_divergence(p, q) <= 1.0


# ---------------------------------------------------------------- decoding
def test_top_k_keeps_exactly_k():
    out = top_k_filter([1.0, 5.0, 3.0, 2.0, 4.0], k=2)
    finite = [x for x in out if math.isfinite(x)]
    assert len(finite) == 2
    assert set(finite) == {5.0, 4.0}


def test_top_k_larger_than_n_is_identity():
    xs = [1.0, 2.0]
    assert top_k_filter(xs, k=10) == xs


def test_top_p_keeps_enough_mass():
    logits = [3.0, 2.0, 1.0, 0.0, -1.0]
    probs = softmax(logits)
    masked = top_p_filter(logits, p=0.9)
    kept = sum(probs[i] for i, v in enumerate(masked) if math.isfinite(v))
    assert kept >= 0.9 - 1e-9


# ------------------------------------------------------- uncertainty / stats
def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0


def test_wilson_shrinks_with_n():
    _, hi1 = wilson_interval(8, 10)
    _, hi2 = wilson_interval(800, 1000)
    assert hi2 < hi1


def test_ece_zero_for_perfect_calibration():
    # perfect calibration: inside every bin the mean confidence equals accuracy
    conf = [0.0, 0.0, 1.0, 1.0]
    corr = [0, 0, 1, 1]
    assert expected_calibration_error(conf, corr, n_bins=2) == pytest.approx(0.0, abs=1e-12)


def test_ece_bounded():
    conf = [0.95] * 10
    corr = [0] * 10
    ece = expected_calibration_error(conf, corr, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert ece > 0.5


def test_bootstrap_ci_brackets_mean():
    rng = DeterministicRNG(3)
    xs = [float(i % 7) for i in range(60)]
    lo, hi = bootstrap_mean_ci(xs, rng, n_boot=400)
    assert lo <= statistics.mean(xs) <= hi


def test_beta_sample_mean():
    rng = DeterministicRNG(11)
    draws = [beta_sample(2.0, 5.0, rng) for _ in range(4000)]
    assert all(0.0 <= d <= 1.0 for d in draws)
    assert statistics.mean(draws) == pytest.approx(2 / 7, abs=0.02)


def test_ucb_unvisited_is_infinite():
    assert ucb_score(0.0, 0, 10) == math.inf
    assert ucb_score(1.0, 5, 10) > 1.0


# ------------------------------------------------------------------ the RNG
def test_rng_is_deterministic():
    a = [DeterministicRNG(42).random() for _ in range(5)]
    b = [DeterministicRNG(42).random() for _ in range(5)]
    assert a == b


def test_rng_fork_is_stable_and_distinct():
    r1 = DeterministicRNG(1).fork(7)
    r2 = DeterministicRNG(1).fork(7)
    r3 = DeterministicRNG(1).fork(8)
    assert [r1.random() for _ in range(3)] == [r2.random() for _ in range(3)]
    assert r1.random() != r3.random()


def test_rng_randbelow_in_range():
    rng = DeterministicRNG(5)
    vals = [rng.randbelow(4) for _ in range(200)]
    assert all(0 <= v < 4 for v in vals)
    assert len(set(vals)) == 4
