"""
L6 — Baseline inference-time strategies (基线策略).

Implemented exactly as published, so comparisons are honest:

* ``greedy``             — one sample, no aggregation (the naive baseline)
* ``self_consistency``   — Wang et al. 2022: sample n, majority vote, one vote per chain
* ``best_of_n_verified`` — sample n, keep the highest verification-trust chain
* ``vwsc``               — ★ this project's flagship (see :mod:`aggregate.vwsc`)
* ``vwsc_adaptive``      — VWSC + adaptive compute budgeting (L7)

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from typing import Dict, List

from ..core.budget import state_from_distribution
from ..core.trace import normalize_answer
from .base import SolveContext, Strategy, StrategyResult

__all__ = ["Greedy", "SelfConsistency", "BestOfNVerified", "VWSCStrategy", "VWSCAdaptive"]


class Greedy(Strategy):
    """Single sample, first answer wins."""

    name = "greedy"

    def solve(self, ctx: SolveContext) -> StrategyResult:
        traces = ctx.sample(1, salt=1)
        agg = self._aggregate(ctx, traces)
        return self._finish(ctx, traces, agg, "single-sample")


class SelfConsistency(Strategy):
    """Wang et al. 2022 — majority vote, verification ignored."""

    name = "self_consistency"

    def solve(self, ctx: SolveContext) -> StrategyResult:
        n = ctx.budget.cfg.max_traces
        traces = ctx.sample(n, salt=2)
        agg = self._aggregate(ctx, traces)
        # deliberately take the *majority* answer, not the verified one
        agg = dict(agg)
        agg["answer"] = agg["majority_answer"]
        agg["confidence"] = agg["majority_share"]
        return self._finish(ctx, traces, agg, "fixed-budget")


class BestOfNVerified(Strategy):
    """Sample n, return the answer of the highest-trust chain (verifier as selector)."""

    name = "best_of_n_verified"

    def solve(self, ctx: SolveContext) -> StrategyResult:
        n = ctx.budget.cfg.max_traces
        traces = ctx.sample(n, salt=3)
        self._verify(ctx, traces)  # warm the verifier; scores come from _aggregate
        agg = self._aggregate(ctx, traces)
        scores = agg.get("scores", [])
        if scores:
            best = max(scores, key=lambda s: (s.trust, s.trace_id))
            by_id = {t.id: t for t in traces}
            cand = by_id.get(best.trace_id)
            if cand is not None:
                agg["answer"] = cand.answer
                agg["confidence"] = best.trust
        return self._finish(ctx, traces, agg, "fixed-budget")

    def _verify(self, ctx: SolveContext, traces: List):
        from .base import verify_all

        return verify_all(traces, ctx.registry)


class VWSCStrategy(Strategy):
    """★ Verification-Weighted Self-Consistency with a fixed budget."""

    name = "vwsc"

    def solve(self, ctx: SolveContext) -> StrategyResult:
        n = ctx.budget.cfg.max_traces
        traces = ctx.sample(n, salt=4)
        agg = self._aggregate(ctx, traces)
        return self._finish(ctx, traces, agg, "fixed-budget")


class VWSCAdaptive(Strategy):
    """★ VWSC + adaptive early stopping (L7 budget scheduler).

    Samples in geometrically growing batches and stops as soon as the
    verification-weighted distribution is decisive. Typically uses 2-4x fewer
    samples than fixed-budget VWSC at equal or better accuracy.
    """

    name = "vwsc_adaptive"

    def solve(self, ctx: SolveContext) -> StrategyResult:
        traces: List = []
        iteration = 0
        stop = "exhausted"
        agg: Dict[str, object] = {}
        while True:
            batch = ctx.budget.next_batch(iteration, len(traces))
            if batch <= 0:
                stop = "exhausted"
                break
            new = ctx.sample(batch, salt=100 + iteration)
            traces.extend(new)
            ctx.budget.record(batch)
            agg = self._aggregate(ctx, traces)

            dist: Dict[str, float] = dict(agg.get("answer_distribution", {}))
            support: Dict[str, int] = {}
            for t in traces:
                k = normalize_answer(t.answer)
                support[k] = support.get(k, 0) + 1
            state = state_from_distribution(
                n_sampled=len(traces),
                n_kept=int(agg.get("n_kept", 0)),
                n_gated=int(agg.get("n_gated", 0)),
                dist=dist,
                mean_trust=float(agg.get("mean_trust", 0.0)),
                support=support,
            )
            done, reason = ctx.budget.should_stop(state)
            if done:
                stop = reason
                break
            iteration += 1
            if iteration > 64:  # hard safety valve
                stop = "exhausted"
                break
        return self._finish(
            ctx,
            traces,
            agg,
            stop,
            extra={"batches": list(ctx.budget.history), "budget": ctx.budget.as_dict()},
        )
