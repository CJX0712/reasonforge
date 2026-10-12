"""
L5 — VWSC: Verification-Weighted Self-Consistency (验证加权自一致性). ★ 旗舰算法 ★

Why plain self-consistency fails
--------------------------------
Wang et al. (2022) majority-vote is *blind*: it gives one vote to every chain.
But an LLM's typical failure is a **locally consistent, globally wrong** chain —
one bad step, then every later step correctly derived from that bad premise.
Such a chain looks flawless to a reader and to a vote counter. It is, however,
exactly what a claim-level verifier can catch.

What VWSC adds
--------------
1. **Dependency-aware trust propagation.** A claim whose *premises* were
   refuted is discounted even if it itself verifies::

       eff(c) = base(c) * Π_{d ∈ deps(c)} (α + (1-α)·eff(d)),   α = 0.5

   This is a Bayesian-flavoured "garbage in, discounted out" rule. It is what
   separates VWSC from simply counting failed claims.

2. **Depth-weighted aggregation.** Downstream claims carry more weight — an
   error at step 4 poisons the answer more than an error at step 1 that gets
   corrected later::

       w(c) = 1 + β·depth(c)

3. **Hard gate.** Chains whose trust score falls below ``gate`` are dropped
   entirely rather than merely down-weighted (a refuted chain should not get a
   vote at all).

4. **Verification-weighted vote.** Remaining chains vote with weight::

       W_i = exp(γ · v_i) · softmax(logprob)_i

Invariants (checked in ``tests/test_vwsc.py``)
----------------------------------------------
* all-clean traces -> VWSC == majority vote (no harm when nothing is wrong)
* a trace with a refuted claim scores strictly below an otherwise identical
  clean trace
* weights are non-negative and sum to 1
* the gate is monotone: raising ``gate`` never *adds* a chain

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.numeric import entropy, softmax
from ..core.trace import Claim, Trace, normalize_answer
from ..verifiers.base import VerificationReport, VerificationResult

__all__ = [
    "VWSCConfig",
    "TraceScore",
    "score_trace",
    "score_traces",
    "vwsc_aggregate",
    "majority_vote",
    "weighted_vote",
    "UNKNOWN_TRUST",
]

UNKNOWN_TRUST = 0.5


@dataclass
class VWSCConfig:
    """Hyper-parameters of the aggregator."""

    alpha: float = 0.5      # dependency discount floor: eff *= alpha + (1-alpha)*parent
    beta: float = 0.5       # depth weight: w = 1 + beta*depth
    gamma: float = 3.0      # trust -> vote-weight exponent
    gate: float = 0.55      # hard filter on trust score
    unknown_trust: float = UNKNOWN_TRUST
    use_logprob: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha in [0,1]")
        if self.beta < 0:
            raise ValueError("beta >= 0")
        if self.gamma < 0:
            raise ValueError("gamma >= 0")
        if not 0.0 <= self.gate <= 1.0:
            raise ValueError("gate in [0,1]")


@dataclass
class TraceScore:
    """Per-trace diagnostics produced by :func:`score_trace`."""

    trace_id: str
    trust: float
    n_passed: int = 0
    n_failed: int = 0
    n_unknown: int = 0
    n_claims: int = 0
    gated: bool = False
    weight: float = 0.0
    claim_trust: Dict[str, float] = field(default_factory=dict)
    counterexamples: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "trust": self.trust,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_unknown": self.n_unknown,
            "n_claims": self.n_claims,
            "gated": self.gated,
            "weight": self.weight,
            "counterexamples": list(self.counterexamples),
        }


def _base_trust(res: VerificationResult, unknown: float) -> float:
    if res.failed:
        return 0.0
    if res.passed:
        return 0.5 + 0.5 * res.confidence  # confidence 1.0 -> 1.0
    return unknown * res.confidence if res.confidence > 0 else 0.0


def score_trace(
    trace: Trace,
    report: VerificationReport,
    cfg: Optional[VWSCConfig] = None,
) -> TraceScore:
    """Compute dependency-aware trust for one trace.

    Returns :class:`TraceScore` with ``trust`` in [0,1] (1 == fully verified).
    """
    cfg = cfg or VWSCConfig()
    trace.validate()
    try:
        order = trace.topo_order()
    except Exception:
        order = trace.claims()

    cmap = {c.id: c for c in trace.claims()}
    depths: Dict[str, int] = {}
    eff: Dict[str, float] = {}

    for c in order:
        if not c.deps:
            depths[c.id] = 0
        else:
            depths[c.id] = 1 + max(depths.get(d, 0) for d in c.deps)
        res = report.results.get(c.id)
        base = _base_trust(res, cfg.unknown_trust) if res is not None else 0.0
        discount = 1.0
        for d in c.deps:
            parent = eff.get(d, cfg.unknown_trust)
            discount *= cfg.alpha + (1.0 - cfg.alpha) * parent
        if c.deps:
            base *= discount
        eff[c.id] = min(1.0, max(0.0, base))

    if not eff:
        return TraceScore(trace_id=trace.id, trust=0.0, gated=True)

    num = 0.0
    den = 0.0
    for cid, t in eff.items():
        w = 1.0 + cfg.beta * depths.get(cid, 0)
        num += w * t
        den += w
    trust = num / den if den > 0 else 0.0

    failed = [r for r in report.results.values() if r.failed]
    unknown = [r for r in report.results.values() if r.unknown]
    passed = [r for r in report.results.values() if r.passed]
    return TraceScore(
        trace_id=trace.id,
        trust=trust,
        n_passed=len(passed),
        n_failed=len(failed),
        n_unknown=len(unknown),
        n_claims=len(eff),
        gated=trust < cfg.gate,
        weight=0.0,
        claim_trust=eff,
        counterexamples=[r.counterexample for r in failed if r.counterexample],
    )


def score_traces(
    traces: Sequence[Trace],
    reports: Dict[str, VerificationReport],
    cfg: Optional[VWSCConfig] = None,
) -> List[TraceScore]:
    """Score every trace and attach normalised vote weights."""
    cfg = cfg or VWSCConfig()
    scores = [score_trace(t, reports[t.id], cfg) for t in traces if t.id in reports]
    alive = [s for s in scores if not s.gated]
    alive = alive or scores  # never drop everything: fall back to all

    priors: Dict[str, float] = {}
    if cfg.use_logprob:
        lp = [t.logprob for t in traces if t.id in reports]
        ps = softmax(lp, temperature=1.0)
        priors = {t.id: p for t, p in zip([t for t in traces if t.id in reports], ps)}
    else:
        n = len(alive) or 1
        priors = {s.trace_id: 1.0 / n for s in alive}

    raw: Dict[str, float] = {}
    for s in alive:
        p = priors.get(s.trace_id, 1.0)
        raw[s.trace_id] = math.exp(cfg.gamma * (s.trust - 1.0)) * p
    total = sum(raw.values())
    for s in scores:
        s.weight = (raw[s.trace_id] / total) if (s.trace_id in raw and total > 0) else 0.0
    return scores


def weighted_vote(traces: Sequence[Trace], weights: Sequence[float]) -> Tuple[str, float]:
    """Weighted vote over canonical answer keys.

    Returns ``(best_answer_key, share)`` where ``share`` in [0,1] is the winning
    answer's fraction of total mass — a natural confidence proxy.
    """
    if len(traces) != len(weights):
        raise ValueError("length mismatch")
    mass: Dict[str, float] = {}
    display: Dict[str, str] = {}
    for t, w in zip(traces, weights):
        if w <= 0:
            continue
        k = normalize_answer(t.answer)
        mass[k] = mass.get(k, 0.0) + float(w)
        display.setdefault(k, t.answer)
    if not mass:
        return "", 0.0
    best = max(mass.items(), key=lambda kv: (kv[1], kv[0]))
    total = sum(mass.values())
    return display.get(best[0], best[0]), (best[1] / total if total > 0 else 0.0)


def majority_vote(traces: Sequence[Trace]) -> Tuple[str, float]:
    """Plain self-consistency baseline (Wang et al. 2022): one chain, one vote."""
    counts: Dict[str, int] = {}
    display: Dict[str, str] = {}
    for t in traces:
        k = normalize_answer(t.answer)
        counts[k] = counts.get(k, 0) + 1
        display.setdefault(k, t.answer)
    if not counts:
        return "", 0.0
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    total = sum(counts.values())
    return display.get(best[0], best[0]), best[1] / total


def vwsc_aggregate(
    traces: Sequence[Trace],
    reports: Dict[str, VerificationReport],
    cfg: Optional[VWSCConfig] = None,
) -> Dict[str, object]:
    """Full VWSC pipeline: score -> gate -> weighted vote -> diagnostics."""
    cfg = cfg or VWSCConfig()
    scores = score_traces(traces, reports, cfg)
    kept = [t for t, s in zip([t for t in traces if t.id in reports], scores) if not s.gated]
    kept = kept or [t for t in traces if t.id in reports]
    ws = [s.weight for s, t in zip(scores, [t for t in traces if t.id in reports])]

    answer_v, share_v = weighted_vote(
        [t for t in traces if t.id in reports], ws
    )
    answer_m, share_m = majority_vote(traces)

    # fallback: if everything was gated, take the highest-trust chain's answer
    if not answer_v and scores:
        best = max(scores, key=lambda s: (s.trust, s.trace_id))
        by_id = {t.id: t for t in traces}
        answer_v = by_id[best.trace_id].answer if best.trace_id in by_id else answer_m
        share_v = best.trust

    dist: Dict[str, float] = {}
    for t, w in zip([t for t in traces if t.id in reports], ws):
        dist[normalize_answer(t.answer)] = dist.get(normalize_answer(t.answer), 0.0) + w
    H = entropy(list(dist.values()), base=2.0)

    return {
        "answer": answer_v,
        "confidence": share_v,
        "majority_answer": answer_m,
        "majority_share": share_m,
        "disagreement_entropy": H,
        "n_traces": len(traces),
        "n_kept": len(kept),
        "n_gated": sum(1 for s in scores if s.gated),
        "mean_trust": (sum(s.trust for s in scores) / len(scores)) if scores else 0.0,
        "scores": scores,
        "answer_distribution": dist,
    }
