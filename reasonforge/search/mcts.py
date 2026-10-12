"""
L6 — UCB decoding-policy bandit (UCB 解码策略臂).

A different axis than DVTS: instead of diversifying *chains*, diversify the
**decoding policy**. Each arm is a sampling configuration (temperature / top_p);
each round we pull an arm with UCB1, sample one chain, and use that chain's
verification trust as the reward.

This is the honest way to spend a budget when you do not know in advance whether
the task wants conservative or exploratory decoding — and the verifier gives us a
free reward signal (no labels needed at test time).

Reward design: ``r = trust`` (in [0,1]) so UCB1's bounded-reward assumption holds.

Invariant: every arm is pulled at least once before UCB is used (cold start), and
the final answer always comes from VWSC over *all* sampled chains — the bandit
only steers where the samples go.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..aggregate.vwsc import score_trace
from ..core.numeric import ucb_score
from ..core.trace import Trace
from .base import SolveContext, Strategy, StrategyResult, verify_all

__all__ = ["UCBPolicyBandit", "ARMS"]


ARMS: Tuple[Tuple[float, float], ...] = (
    (0.3, 0.90),
    (0.7, 0.95),
    (1.0, 0.98),
    (1.3, 1.00),
)


@dataclass
class UCBPolicyBandit(Strategy):
    arms: Sequence[Tuple[float, float]] = ARMS
    c: float = 1.414
    name: str = "ucb_policy_bandit"
    stats: Dict[int, List[float]] = field(default_factory=dict)
    pulls: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("at least one arm")
        self.stats = {i: [] for i in range(len(self.arms))}
        self.pulls = {i: 0 for i in range(len(self.arms))}

    def _select(self, t: int) -> int:
        # cold start: pull each arm once in order
        for i in range(len(self.arms)):
            if self.pulls[i] == 0:
                return i
        best_i, best_v = 0, -1.0
        for i in range(len(self.arms)):
            rewards = self.stats[i]
            mean = sum(rewards) / len(rewards) if rewards else 0.0
            v = ucb_score(mean, self.pulls[i], t, c=self.c)
            if v > best_v or (v == best_v and i < best_i):
                best_i, best_v = i, v
        return best_i

    def solve(self, ctx: SolveContext) -> StrategyResult:
        total = ctx.budget.cfg.max_traces
        traces: List[Trace] = []
        history: List[Dict[str, object]] = []

        for t in range(1, total + 1):
            arm = self._select(t)
            temp, topp = self.arms[arm]
            batch = ctx.sample(1, salt=400 + t, temperature=temp, top_p=topp)
            if not batch:
                continue
            tr = batch[0]
            traces.append(tr)
            try:
                tr.validate()
                rep = ctx.registry.verify_trace(tr)
                trust = score_trace(tr, rep, ctx.vwsc).trust
            except Exception:
                trust = 0.0
            self.stats[arm].append(trust)
            self.pulls[arm] += 1
            history.append({"t": t, "arm": arm, "temp": temp, "trust": trust})

        agg = self._aggregate(ctx, traces)
        means = {
            i: (sum(v) / len(v) if v else 0.0) for i, v in self.stats.items()
        }
        return self._finish(
            ctx,
            traces,
            agg,
            "fixed-budget",
            extra={
                "arm_pulls": dict(self.pulls),
                "arm_mean_reward": means,
                "history": history,
            },
        )
