"""
L7 — Adaptive compute budgeting (自适应算力预算调度).

Test-time compute is a *resource*, and spending it uniformly is wasteful: easy
questions are settled by 4 samples, hard ones need 64. This module decides **when
to stop sampling** using signals that only a verification layer can provide.

Stopping rules (evaluated after every batch)
--------------------------------------------
1. ``confident``  — the leading answer holds ``>= confidence_target`` of the
   verification-weighted mass **and** at least ``min_verified`` ungated traces
   support it.
2. ``consensus``  — normalised answer entropy below ``entropy_floor`` (chains
   agree; more samples cannot change the outcome).
3. ``exhausted``  — ``max_traces`` reached.
4. ``starved``    — every chain so far was gated out and we hit the ceiling:
   verification cannot rescue this one, return the best-trust chain.

Batch sizes grow geometrically (``initial_batch * growth^k``) and are clipped to
the remaining budget, so easy items are cheap and hard ones get the full budget.

Invariant: :meth:`plan` never emits a total larger than ``max_traces``, and its
sequence is deterministic.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .numeric import entropy, normalized_entropy

__all__ = [
    "BudgetConfig",
    "BudgetScheduler",
    "BudgetState",
    "STOP_REASONS",
]

STOP_REASONS = ("confident", "consensus", "exhausted", "starved", "continue")


@dataclass
class BudgetConfig:
    max_traces: int = 32
    initial_batch: int = 8
    growth: float = 1.6
    confidence_target: float = 0.70
    entropy_floor: float = 0.20
    min_verified: int = 2

    def __post_init__(self) -> None:
        if self.max_traces < 1:
            raise ValueError("max_traces >= 1")
        if self.initial_batch < 1:
            raise ValueError("initial_batch >= 1")
        if self.growth < 1.0:
            raise ValueError("growth >= 1")
        if not 0.0 < self.confidence_target <= 1.0:
            raise ValueError("confidence_target in (0,1]")
        if not 0.0 <= self.entropy_floor <= 1.0:
            raise ValueError("entropy_floor in [0,1]")


@dataclass
class BudgetState:
    """Snapshot handed to the scheduler after each batch."""

    n_sampled: int = 0
    n_kept: int = 0
    n_gated: int = 0
    top_share: float = 0.0
    answer_entropy: float = 1.0
    mean_trust: float = 0.0
    top_supported_by: int = 0


class BudgetScheduler:
    """Decides batch sizes and early-stopping."""

    def __init__(self, cfg: Optional[BudgetConfig] = None) -> None:
        self.cfg = cfg or BudgetConfig()
        self.history: List[int] = []

    # -- planning ---------------------------------------------------------
    def plan(self) -> List[int]:
        """The full batch schedule (deterministic, total <= max_traces)."""
        out: List[int] = []
        remaining = self.cfg.max_traces
        n = float(self.cfg.initial_batch)
        while remaining > 0:
            take = int(max(1, math.floor(n)))
            take = min(take, remaining)
            out.append(take)
            remaining -= take
            n *= self.cfg.growth
        return out

    def next_batch(self, iteration: int, n_sampled: int) -> int:
        schedule = self.plan()
        if iteration >= len(schedule):
            return 0
        remaining = self.cfg.max_traces - n_sampled
        return max(0, min(schedule[iteration], remaining))

    # -- stopping ---------------------------------------------------------
    def should_stop(self, state: BudgetState) -> Tuple[bool, str]:
        cfg = self.cfg
        if state.n_sampled >= cfg.max_traces:
            return True, "exhausted"
        if state.n_kept == 0 and state.n_sampled >= cfg.initial_batch:
            # nothing survives verification -> more of the same won't help
            if state.n_sampled >= cfg.max_traces // 2:
                return True, "starved"
        if (
            state.top_share >= cfg.confidence_target
            and state.top_supported_by >= cfg.min_verified
            and state.n_kept >= cfg.min_verified
        ):
            return True, "confident"
        if state.n_sampled >= cfg.initial_batch and state.answer_entropy <= cfg.entropy_floor:
            return True, "consensus"
        return False, "continue"

    # -- bookkeeping ------------------------------------------------------
    def record(self, batch_size: int) -> None:
        self.history.append(batch_size)

    def total_used(self) -> int:
        return sum(self.history)

    def as_dict(self) -> Dict[str, object]:
        return {
            "config": {
                "max_traces": self.cfg.max_traces,
                "initial_batch": self.cfg.initial_batch,
                "growth": self.cfg.growth,
                "confidence_target": self.cfg.confidence_target,
                "entropy_floor": self.cfg.entropy_floor,
                "min_verified": self.cfg.min_verified,
            },
            "history": list(self.history),
            "total_used": self.total_used(),
        }


def state_from_distribution(
    n_sampled: int,
    n_kept: int,
    n_gated: int,
    dist: Dict[str, float],
    mean_trust: float,
    support: Dict[str, int],
) -> BudgetState:
    """Build a :class:`BudgetState` from an answer distribution."""
    if dist:
        total = sum(dist.values()) or 1.0
        top = max(dist.items(), key=lambda kv: (kv[1], kv[0]))
        share = top[1] / total
        H = normalized_entropy(list(dist.values()), base=2.0)
    else:
        share, H = 0.0, 1.0
    return BudgetState(
        n_sampled=n_sampled,
        n_kept=n_kept,
        n_gated=n_gated,
        top_share=share,
        answer_entropy=H,
        mean_trust=mean_trust,
        top_supported_by=support.get(max(dist, key=lambda k: (dist[k], k)), 0) if dist else 0,
    )
