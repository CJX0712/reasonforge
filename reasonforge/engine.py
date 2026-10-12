"""
Top-level orchestration (顶层编排).

:class:`ReasonForge` is the single entry point: give it a task, get back an
answer with a confidence, a full audit trail (every trace, every verification
result, every counterexample) and — when CEGAR is enabled — the repair history.

Typical use::

    from reasonforge import ReasonForge
    from reasonforge.bench.tasks import get_task

    rf = ReasonForge(seed=42)
    out = rf.solve(get_task("arith-01"), strategy="vwsc_adaptive", use_cegar=True)
    print(out.answer, out.confidence, out.correct)

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .aggregate.vwsc import VWSCConfig, vwsc_aggregate
from .bench.runner import BenchmarkReport, Runner
from .bench.tasks import TASKS, Task
from .core.budget import BudgetConfig, BudgetScheduler
from .core.numeric import DeterministicRNG
from .core.trace import Trace, normalize_answer
from .llm.base import LLMBackend
from .llm.mock import MockStudent
from .refine.cegar import CegarConfig, CegarRefinement, cegar_refine
from .search import STRATEGIES, make_strategy
from .search.base import SolveContext, StrategyResult, verify_all
from .verifiers import VerifierRegistry, default_registry
from .verifiers.base import VerificationReport

__all__ = ["ReasonForge", "SolveResult"]


@dataclass
class SolveResult:
    answer: str = ""
    confidence: float = 0.0
    strategy: str = ""
    n_samples: int = 0
    stop_reason: str = ""
    mean_trust: float = 0.0
    n_gated: int = 0
    correct: Optional[bool] = None
    traces: List[Trace] = field(default_factory=list)
    reports: Dict[str, VerificationReport] = field(default_factory=dict)
    cegar: Optional[CegarRefinement] = None
    detail: Dict[str, object] = field(default_factory=dict)

    def counterexamples(self) -> List[str]:
        out: List[str] = []
        for rep in self.reports.values():
            out.extend(rep.counterexamples())
        return out

    def as_dict(self) -> Dict[str, object]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "n_samples": self.n_samples,
            "stop_reason": self.stop_reason,
            "mean_trust": self.mean_trust,
            "n_gated": self.n_gated,
            "correct": self.correct,
            "cegar": self.cegar.as_dict() if self.cegar else None,
            "detail": self.detail,
        }


class ReasonForge:
    """The engine: sample -> verify -> aggregate -> (refine) -> answer."""

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        registry: Optional[VerifierRegistry] = None,
        seed: int = 0,
        budget: Optional[BudgetConfig] = None,
        vwsc: Optional[VWSCConfig] = None,
        cegar: Optional[CegarConfig] = None,
    ) -> None:
        self.backend = backend or MockStudent()
        self.registry = registry or default_registry()
        self.seed = seed
        self.budget_cfg = budget or BudgetConfig()
        self.vwsc_cfg = vwsc or VWSCConfig()
        self.cegar_cfg = cegar or CegarConfig()
        self.runner = Runner(
            backend=self.backend,
            registry=self.registry,
            seed=seed,
            budget=self.budget_cfg,
            vwsc=self.vwsc_cfg,
        )

    # ------------------------------------------------------------------
    def make_ctx(self, task: Task, salt: int = 0) -> SolveContext:
        return SolveContext(
            task=task,
            backend=self.backend,
            rng=DeterministicRNG(self.seed).fork(salt),
            registry=self.registry,
            budget=BudgetScheduler(self.budget_cfg),
            vwsc=self.vwsc_cfg,
        )

    def solve(
        self,
        task: Task,
        strategy: str = "vwsc_adaptive",
        use_cegar: bool = True,
        salt: int = 0,
        cegar_cfg: Optional[CegarConfig] = None,
    ) -> SolveResult:
        """Solve one task end-to-end with a full audit trail."""
        strat = make_strategy(strategy)
        ctx = self.make_ctx(task, salt)
        res: StrategyResult = strat.solve(ctx)

        traces = res.traces
        agg: Dict[str, object] = {}
        ref: Optional[CegarRefinement] = None
        stop = res.stop_reason

        need_refine = (
            use_cegar
            and self.backend.supports_feedback()
            and (
                float(res.confidence) < (cegar_cfg or self.cegar_cfg).target_confidence
                or res.detail.get("mean_trust", 1.0) < (cegar_cfg or self.cegar_cfg).min_trust
            )
        )
        if need_refine:
            ref = cegar_refine(ctx, traces, cegar_cfg or self.cegar_cfg)
            traces = ref.traces
            reports = verify_all(traces, ctx.registry)
            agg = vwsc_aggregate([t for t in traces if t.id in reports], reports, ctx.vwsc)
            stop = f"{res.stop_reason}+cegar:{ref.stop_reason}"
        else:
            reports = verify_all(traces, ctx.registry)
            agg = vwsc_aggregate([t for t in traces if t.id in reports], reports, ctx.vwsc)

        answer = str(agg.get("answer", res.answer))
        correct: Optional[bool] = None
        if getattr(task, "gold_answer", None) is not None:
            correct = normalize_answer(task.gold_answer) == normalize_answer(answer)

        return SolveResult(
            answer=answer,
            confidence=float(agg.get("confidence", res.confidence)),
            strategy=strategy,
            n_samples=len(traces),
            stop_reason=stop,
            mean_trust=float(agg.get("mean_trust", 0.0)),
            n_gated=int(agg.get("n_gated", 0)),
            correct=correct,
            traces=traces,
            reports=reports,
            cegar=ref,
            detail=dict(res.detail),
        )

    # ------------------------------------------------------------------
    def benchmark(
        self,
        strategies: Sequence[str] = ("greedy", "self_consistency", "best_of_n_verified", "vwsc", "vwsc_adaptive"),
        tasks: Optional[Sequence[Task]] = None,
        salt: int = 0,
    ) -> Dict[str, BenchmarkReport]:
        """Compare strategies over the built-in task set."""
        return self.runner.run(list(strategies), list(tasks or TASKS), salt=salt)

    # ------------------------------------------------------------------
    def explain(self, result: SolveResult, limit: int = 12) -> str:
        """Human-readable audit trail — every verdict, every counterexample."""
        lines: List[str] = []
        lines.append(f"strategy      : {result.strategy}")
        lines.append(f"answer        : {result.answer}  (confidence {result.confidence:.3f})")
        lines.append(f"samples       : {result.n_samples}   stop={result.stop_reason}")
        lines.append(f"mean trust    : {result.mean_trust:.3f}  gated={result.n_gated}")
        if result.correct is not None:
            lines.append(f"correct       : {result.correct}")
        lines.append("")
        lines.append("per-trace verification:")
        for t in result.traces[:limit]:
            rep = result.reports.get(t.id)
            if rep is None:
                lines.append(f"  {t.id}: <unverifiable>")
                continue
            lines.append(
                f"  {t.id:<22} ans={t.answer:<12} "
                f"P={rep.n_passed()} F={rep.n_failed()} U={rep.n_unknown()}"
            )
            for r in rep.failed()[:3]:
                ce = f"  [{r.counterexample}]" if r.counterexample else ""
                lines.append(f"      ✗ {r.claim_id}: {r.detail}{ce}")
        if result.cegar is not None:
            lines.append("")
            lines.append(f"CEGAR rounds  : {result.cegar.n_rounds} ({result.cegar.stop_reason})")
            for i, fb in enumerate(result.cegar.feedback_used):
                lines.append(f"  round {i}: {len(fb)} counterexamples")
                for f in fb[:3]:
                    lines.append(f"      {f.render()}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ReasonForge backend={self.backend.name!r} seed={self.seed}>"
