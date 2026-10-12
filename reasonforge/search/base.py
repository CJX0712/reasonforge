"""
L3/L6 — Inference-time strategy contract (推理时策略契约).

Every strategy consumes a :class:`SolveContext` and returns a
:class:`StrategyResult`. Strategies differ only in **how they spend compute**;
they all share the same verification layer, which is what makes them comparable.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..aggregate.vwsc import VWSCConfig, vwsc_aggregate
from ..bench.tasks import Task
from ..core.budget import BudgetScheduler
from ..core.numeric import DeterministicRNG
from ..core.trace import Trace
from ..llm.base import GenerationConfig, LLMBackend
from ..verifiers import VerifierRegistry, default_registry
from ..verifiers.base import VerificationReport

__all__ = ["StrategyResult", "SolveContext", "Strategy", "verify_all"]


@dataclass
class StrategyResult:
    answer: str = ""
    confidence: float = 0.0
    n_samples: int = 0
    strategy: str = ""
    stop_reason: str = ""
    detail: Dict[str, object] = field(default_factory=dict)
    traces: List[Trace] = field(default_factory=list)
    reports: Dict[str, VerificationReport] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "n_samples": self.n_samples,
            "strategy": self.strategy,
            "stop_reason": self.stop_reason,
            "detail": self.detail,
        }


@dataclass
class SolveContext:
    """Everything a strategy needs. Fully deterministic given ``rng``."""

    task: Task
    backend: LLMBackend
    rng: DeterministicRNG
    registry: VerifierRegistry = field(default_factory=default_registry)
    budget: BudgetScheduler = field(default_factory=BudgetScheduler)
    vwsc: VWSCConfig = field(default_factory=VWSCConfig)
    gen: GenerationConfig = field(default_factory=GenerationConfig)

    def sample(self, n: int, salt: int = 0, **gen_overrides) -> List[Trace]:
        cfg = GenerationConfig(
            n=n,
            temperature=gen_overrides.get("temperature", self.gen.temperature),
            top_p=gen_overrides.get("top_p", self.gen.top_p),
            max_steps=self.gen.max_steps,
        )
        return self.backend.sample(self.task, cfg, self.rng.fork(salt))


def verify_all(
    traces: List[Trace], registry: VerifierRegistry
) -> Dict[str, VerificationReport]:
    """Verify every trace, skipping (and reporting) structurally broken ones."""
    out: Dict[str, VerificationReport] = {}
    for t in traces:
        try:
            t.validate()
        except Exception:
            continue
        out[t.id] = registry.verify_trace(t)
    return out


class Strategy(ABC):
    """Abstract inference-time strategy."""

    name: str = "strategy"

    @abstractmethod
    def solve(self, ctx: SolveContext) -> StrategyResult:
        ...

    # -- shared helpers ---------------------------------------------------
    def _aggregate(self, ctx: SolveContext, traces: List[Trace]) -> Dict[str, object]:
        reports = verify_all(traces, ctx.registry)
        return vwsc_aggregate(
            [t for t in traces if t.id in reports], reports, ctx.vwsc
        )

    def _finish(
        self,
        ctx: SolveContext,
        traces: List[Trace],
        agg: Dict[str, object],
        stop_reason: str,
        extra: Optional[Dict[str, object]] = None,
    ) -> StrategyResult:
        reports = verify_all(traces, ctx.registry)
        detail: Dict[str, object] = {
            "mean_trust": agg.get("mean_trust", 0.0),
            "n_kept": agg.get("n_kept", 0),
            "n_gated": agg.get("n_gated", 0),
            "majority_answer": agg.get("majority_answer", ""),
            "majority_share": agg.get("majority_share", 0.0),
            "disagreement_entropy": agg.get("disagreement_entropy", 0.0),
        }
        if extra:
            detail.update(extra)
        return StrategyResult(
            answer=str(agg.get("answer", "")),
            confidence=float(agg.get("confidence", 0.0)),
            n_samples=len(traces),
            strategy=self.name,
            stop_reason=stop_reason,
            detail=detail,
            traces=traces,
            reports=reports,
        )
