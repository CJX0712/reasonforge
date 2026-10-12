"""
L0 — Numerical primitives (数值原语层)。

Zero hard dependency: pure Python standard library.
Optional acceleration via NumPy is *never* required for correctness.

Every function here carries a machine-checkable invariant, verified in
``tests/test_numeric.py`` and documented in ``docs/INVARIANTS.md``.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "logsumexp",
    "softmax",
    "entropy",
    "normalized_entropy",
    "gini_simpson",
    "top_k_filter",
    "top_p_filter",
    "kl_divergence",
    "js_divergence",
    "wilson_interval",
    "bootstrap_mean_ci",
    "expected_calibration_error",
    "beta_sample",
    "ucb_score",
    "DeterministicRNG",
    "safe_div",
    "clamp",
    "almost_equal",
]

_EPS = 1e-300
_LOG2 = math.log(2.0)


# --------------------------------------------------------------------------
# Stable transforms
# --------------------------------------------------------------------------
def logsumexp(values: Iterable[float]) -> float:
    """Numerically stable ``log(sum(exp(x)))``.

    Invariant: ``logsumexp([a]) == a`` and ``logsumexp(x + c) == logsumexp(x) + c``
    (shift equivariance, holds to ~1e-12).
    """
    xs = [float(v) for v in values]
    if not xs:
        return -math.inf
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    total = 0.0
    for x in xs:
        total += math.exp(x - m)
    return m + math.log(total)


def softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    """Temperature-scaled softmax with max-subtraction stabilisation.

    Invariants: all outputs in [0,1]; ``sum == 1`` (<=1e-12); monotone in logits.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    scaled = [float(x) / temperature for x in logits]
    if not scaled:
        return []
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    if s <= 0.0 or math.isinf(s):
        n = len(scaled)
        return [1.0 / n] * n
    return [e / s for e in exps]


def entropy(probs: Iterable[float], base: float = 2.0) -> float:
    """Shannon entropy. Invariant: ``0 <= H <= log_b(n)``; uniform maximises it."""
    ps = [float(p) for p in probs]
    total = sum(ps)
    if total <= 0.0:
        return 0.0
    ps = [p / total for p in ps]
    h = 0.0
    for p in ps:
        if p > _EPS:
            h -= p * math.log(p)
    if base != math.e:
        h /= math.log(base)
    return max(0.0, h)


def normalized_entropy(probs: Iterable[float], base: float = 2.0) -> float:
    """Entropy divided by ``log_b(n)`` -> in [0,1]. Undefined (0.0) for n<=1."""
    ps = list(probs)
    n = len(ps)
    if n <= 1:
        return 0.0
    h_max = math.log(n) / math.log(base)
    if h_max <= 0.0:
        return 0.0
    return min(1.0, max(0.0, entropy(ps, base=base) / h_max))


def gini_simpson(probs: Iterable[float]) -> float:
    """Gini-Simpson index ``1 - sum p_i^2`` — probability that two i.i.d. draws differ.

    Invariant: in [0, 1-1/n]; 0 iff degenerate distribution.
    """
    ps = [float(p) for p in probs]
    total = sum(ps)
    if total <= 0.0:
        return 0.0
    ps = [p / total for p in ps]
    return 1.0 - sum(p * p for p in ps)


def kl_divergence(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """KL(p||q) >= 0, == 0 iff p == q. Supports unnormalised inputs."""
    if len(p) != len(q):
        raise ValueError("length mismatch")
    sp = sum(p)
    sq = sum(q)
    if sp <= 0.0 or sq <= 0.0:
        raise ValueError("distributions must have positive mass")
    out = 0.0
    for pi, qi in zip(p, q):
        pi = pi / sp
        qi = qi / sq
        if pi > _EPS:
            if qi <= _EPS:
                return math.inf
            out += pi * math.log(pi / qi)
    if base != math.e:
        out /= math.log(base)
    return max(0.0, out)


def js_divergence(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Jensen-Shannon divergence: symmetric, bounded by 1 (base 2)."""
    sp = sum(p)
    sq = sum(q)
    pn = [x / sp for x in p]
    qn = [x / sq for x in q]
    m = [0.5 * (a + b) for a, b in zip(pn, qn)]
    return 0.5 * kl_divergence(pn, m, base=base) + 0.5 * kl_divergence(qn, m, base=base)


# --------------------------------------------------------------------------
# Decoding filters
# --------------------------------------------------------------------------
def top_k_filter(logits: Sequence[float], k: int) -> List[float]:
    """Keep the ``k`` largest logits, mask the rest with ``-inf``.

    Invariant: number of finite entries == min(k, n); ordering preserved.
    """
    n = len(logits)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k >= n:
        return [float(x) for x in logits]
    order = sorted(range(n), key=lambda i: (-float(logits[i]), i))
    keep = set(order[:k])
    return [float(logits[i]) if i in keep else -math.inf for i in range(n)]


def top_p_filter(logits: Sequence[float], p: float) -> List[float]:
    """Nucleus (top-p) filtering. Invariant: kept mass >= p (up to float error)."""
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    n = len(logits)
    if n == 0:
        return []
    probs = softmax(logits)
    order = sorted(range(n), key=lambda i: (-probs[i], i))
    out = [-math.inf] * n
    cum = 0.0
    for i in order:
        out[i] = float(logits[i])
        cum += probs[i]
        if cum >= p:
            break
    return out


# --------------------------------------------------------------------------
# Uncertainty quantification
# --------------------------------------------------------------------------
def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Invariant: bounds within [0,1]; lower <= phat <= upper; width shrinks as n grows.
    """
    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def bootstrap_mean_ci(
    values: Sequence[float],
    rng: "DeterministicRNG",
    n_boot: int = 512,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean. Deterministic given ``rng``."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (xs[0], xs[0])
    means: List[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += xs[rng.randbelow(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int(math.floor((alpha / 2.0) * (n_boot - 1)))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * (n_boot - 1)))
    return (means[lo_idx], means[hi_idx])


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[int], n_bins: int = 10
) -> float:
    """ECE = sum_b (n_b/N) * |acc_b - conf_b|.

    Invariant: 0 <= ECE <= 1; == 0 for a perfectly calibrated predictor.
    """
    if len(confidences) != len(correct):
        raise ValueError("length mismatch")
    if not confidences:
        return 0.0
    n = len(confidences)
    bins_acc: List[float] = [0.0] * n_bins
    bins_conf: List[float] = [0.0] * n_bins
    bins_cnt: List[int] = [0] * n_bins
    for c, y in zip(confidences, correct):
        c = min(1.0, max(0.0, float(c)))
        b = int(c * n_bins)
        if b >= n_bins:
            b = n_bins - 1
        bins_acc[b] += float(y)
        bins_conf[b] += c
        bins_cnt[b] += 1
    ece = 0.0
    for b in range(n_bins):
        if bins_cnt[b]:
            ece += (bins_cnt[b] / n) * abs(bins_acc[b] / bins_cnt[b] - bins_conf[b] / bins_cnt[b])
    return min(1.0, max(0.0, ece))


def beta_sample(alpha: float, beta: float, rng: "DeterministicRNG") -> float:
    """Sample Beta(a,b) via two Gamma draws (Marsaglia-Tsang).

    The standard construction X/(X+Y) with X~Gamma(a), Y~Gamma(b) independent
    yields Beta(a,b); the *same* X must appear in numerator and denominator, so
    we draw it once and reuse it.

    Invariant: output in [0,1]; mean converges to a/(a+b) (checked in tests).
    """
    x = _gamma_sample(alpha, rng)
    y = _gamma_sample(beta, rng)
    return x / (x + y + 1e-300)


def _gamma_sample(shape: float, rng: "DeterministicRNG") -> float:
    """Marsaglia-Tsang gamma sampler for shape >= 1; boost for shape < 1."""
    if shape < 1.0:
        return _gamma_sample(shape + 1.0, rng) * (rng.random() ** (1.0 / shape))
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss()
        v = 1.0 + c * x
        if v <= 0.0:
            continue
        v = v * v * v
        u = rng.random()
        if u < 1.0 - 0.0331 * x * x * x * x:
            return d * v
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v


def ucb_score(mean: float, visits: int, total_visits: int, c: float = 1.414) -> float:
    """UCB1 bonus. Invariant: unvisited arms score ``+inf``; monotone in mean."""
    if visits <= 0:
        return math.inf
    if total_visits <= 0:
        return mean
    return mean + c * math.sqrt(math.log(total_visits) / visits)


# --------------------------------------------------------------------------
# Deterministic randomness
# --------------------------------------------------------------------------
class DeterministicRNG:
    """Seeded wrapper over :class:`random.Random` with a stable extra API.

    Guarantees bit-identical replays across CPython versions for the methods
    exposed here (randbelow / gauss / random / shuffle are all stdlib-stable).
    """

    __slots__ = ("_r", "seed")

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._r = random.Random(self.seed)

    def random(self) -> float:
        return self._r.random()

    def randbelow(self, n: int) -> int:
        return self._r.randrange(n)

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._r.gauss(mu, sigma)

    def choice(self, seq: Sequence):
        return seq[self._r.randrange(len(seq))]

    def shuffle(self, xs: List) -> None:
        self._r.shuffle(xs)

    def fork(self, salt: int) -> "DeterministicRNG":
        """Derive an independent child stream (seed mixing via splitmix64-style hash)."""
        h = (self.seed ^ int(salt)) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 31
        return DeterministicRNG(h & 0x7FFFFFFFFFFFFFFF)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DeterministicRNG(seed={self.seed})"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0.0 or math.isnan(b) or math.isinf(b):
        return default
    return a / b


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(x):
        return lo
    return max(lo, min(hi, x))


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def normalize_mapping(d: Mapping[object, float]) -> Dict[object, float]:
    """Normalise a counter into a probability distribution over its keys."""
    total = sum(float(v) for v in d.values())
    if total <= 0.0:
        n = len(d)
        return {k: (1.0 / n if n else 0.0) for k in d}
    return {k: float(v) / total for k, v in d.items()}
