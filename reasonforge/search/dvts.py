"""
L6 — DVTS: Diverse Verifier Tree Search (多样性验证树搜索).

After Beeching et al. (2024), *Scaling test-time compute with verifiers*:
split the budget into ``k`` independent beams, let a verifier pick the best chain
**inside** each beam, then aggregate the beam representatives. The point is
*anti-mode-collapse*: a single global verifier tends to select k near-identical
chains that share the same blind spot; per-beam selection preserves diversity.

We reuse the idea but swap the PRM for our claim-level trust score, which is
dependency-aware (see :mod:`aggregate.vwsc`).

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..core.trace import Trace
from .base import SolveContext, Strategy, StrategyResult, verify_all

__all__ = ["DiverseVerifierTreeSearch", "DVTS"]


@dataclass
class DiverseVerifierTreeSearch(Strategy):
    n_beams: int = 4
    name: str = "dvts"

    def __post_init__(self) -> None:
        if self.n_beams < 1:
            raise ValueError("n_beams >= 1")

    def solve(self, ctx: SolveContext) -> StrategyResult:
        total = ctx.budget.cfg.max_traces
        k = max(1, min(self.n_beams, total))
        per = max(1, total // k)

        all_traces: List[Trace] = []
        reps: List[Trace] = []
        beam_info: List[Dict[str, object]] = []

        for b in range(k):
            traces = ctx.sample(per, salt=300 + b)
            all_traces.extend(traces)
            reports = verify_all(traces, ctx.registry)
            if not reports:
                beam_info.append({"beam": b, "best_trust": 0.0, "answer": ""})
                continue
            from ..aggregate.vwsc import score_traces

            scores = score_traces(
                [t for t in traces if t.id in reports], reports, ctx.vwsc
            )
            best_score = max(scores, key=lambda s: (s.trust, s.trace_id))
            by_id = {t.id: t for t in traces}
            rep = by_id.get(best_score.trace_id)
            if rep is not None:
                reps.append(rep)
                beam_info.append(
                    {
                        "beam": b,
                        "best_trust": best_score.trust,
                        "answer": rep.answer,
                        "n_failed": best_score.n_failed,
                    }
                )
            else:
                beam_info.append({"beam": b, "best_trust": 0.0, "answer": ""})

        # final aggregation over beam representatives only (that is the point)
        pool = reps or all_traces
        agg = self._aggregate(ctx, pool)
        return self._finish(
            ctx,
            all_traces,
            agg,
            "fixed-budget",
            extra={"n_beams": k, "per_beam": per, "beams": beam_info},
        )


DVTS = DiverseVerifierTreeSearch
