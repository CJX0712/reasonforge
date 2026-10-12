"""
L9 — Benchmark runner (评测运行器).

Compares inference-time strategies on the built-in tasks. Everything is seeded,
so two runs with the same seed produce byte-identical reports.

Correctness is decided by canonical answer matching against the task's gold
answer — never by string equality on raw text.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..aggregate.calibration import calibration_error
from ..aggregate.vwsc import VWSCConfig
from ..core.budget import BudgetConfig, BudgetScheduler
from ..core.numeric import DeterministicRNG
from ..core.trace import normalize_answer
from ..llm.base import LLMBackend
from ..search import STRATEGIES, make_strategy
from ..search.base import SolveContext, StrategyResult
from ..verifiers import VerifierRegistry, default_registry
from .tasks import DOMAINS, TASKS, Task

__all__ = ["TaskResult", "BenchmarkReport", "Runner"]


@dataclass
class TaskResult:
    task_id: str
    domain: str
    gold: str
    predicted: str
    correct: bool
    confidence: float
    n_samples: int
    stop_reason: str = ""
    detail: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "gold": self.gold,
            "predicted": self.predicted,
            "correct": self.correct,
            "confidence": self.confidence,
            "n_samples": self.n_samples,
            "stop_reason": self.stop_reason,
            "detail": self.detail,
        }


@dataclass
class BenchmarkReport:
    strategy: str
    results: List[TaskResult] = field(default_factory=list)

    # -- metrics ----------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        return (sum(1 for r in self.results if r.correct) / self.n) if self.n else 0.0

    @property
    def mean_samples(self) -> float:
        return (sum(r.n_samples for r in self.results) / self.n) if self.n else 0.0

    @property
    def mean_confidence(self) -> float:
        return (sum(r.confidence for r in self.results) / self.n) if self.n else 0.0

    def per_domain(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for d in DOMAINS:
            rs = [r for r in self.results if r.domain == d]
            if rs:
                out[d] = sum(1 for r in rs if r.correct) / len(rs)
        return out

    def calibration(self, n_bins: int = 5) -> Dict[str, float]:
        return calibration_error(
            [r.confidence for r in self.results],
            [1 if r.correct else 0 for r in self.results],
            n_bins=n_bins,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "n_tasks": self.n,
            "accuracy": self.accuracy,
            "mean_samples": self.mean_samples,
            "mean_confidence": self.mean_confidence,
            "per_domain": self.per_domain(),
            "calibration": self.calibration(),
            "results": [r.as_dict() for r in self.results],
        }

    def table(self) -> str:
        """Markdown table of headline metrics."""
        dom = self.per_domain()
        cal = self.calibration()
        lines = [
            f"| strategy | accuracy | mean samples | ECE | " + " | ".join(DOMAINS) + " |",
            "|" + "---|" * (4 + len(DOMAINS)),
            "| {} | {:.3f} | {:.1f} | {:.3f} | ".format(
                self.strategy, self.accuracy, self.mean_samples, cal["ece"]
            )
            + " | ".join(f"{dom.get(d, float('nan')):.3f}" for d in DOMAINS)
            + " |",
        ]
        return "\n".join(lines)


class Runner:
    """Runs strategies over tasks, deterministically."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: Optional[VerifierRegistry] = None,
        seed: int = 0,
        budget: Optional[BudgetConfig] = None,
        vwsc: Optional[VWSCConfig] = None,
    ) -> None:
        self.backend = backend
        self.registry = registry or default_registry()
        self.seed = seed
        self.budget_cfg = budget or BudgetConfig()
        self.vwsc_cfg = vwsc or VWSCConfig()

    def make_ctx(self, task: Task, salt: int = 0) -> SolveContext:
        return SolveContext(
            task=task,
            backend=self.backend,
            rng=DeterministicRNG(self.seed).fork(salt),
            registry=self.registry,
            budget=BudgetScheduler(self.budget_cfg),
            vwsc=self.vwsc_cfg,
        )

    def run_task(self, task: Task, strategy_name: str, salt: int = 0) -> TaskResult:
        strat = make_strategy(strategy_name)
        ctx = self.make_ctx(task, salt)
        res: StrategyResult = strat.solve(ctx)
        correct = normalize_answer(task.gold_answer) == normalize_answer(res.answer)
        return TaskResult(
            task_id=task.id,
            domain=task.domain,
            gold=task.gold_answer,
            predicted=res.answer,
            correct=correct,
            confidence=res.confidence,
            n_samples=res.n_samples,
            stop_reason=res.stop_reason,
            detail=res.detail,
        )

    def run(
        self,
        strategies: Sequence[str],
        tasks: Optional[Sequence[Task]] = None,
        salt: int = 0,
    ) -> Dict[str, BenchmarkReport]:
        tasks = list(tasks or TASKS)
        reports: Dict[str, BenchmarkReport] = {}
        for i, name in enumerate(strategies):
            rep = BenchmarkReport(strategy=name)
            for j, task in enumerate(tasks):
                rep.results.append(self.run_task(task, name, salt + i * 1000 + j))
            reports[name] = rep
        return reports


def compare(reports: Dict[str, BenchmarkReport]) -> str:
    """Markdown comparison table across strategies."""
    if not reports:
        return ""
    header = ["| strategy | acc | samples | ECE | " + " | ".join(DOMAINS) + " |"]
    sep = ["|" + "---|" * (4 + len(DOMAINS))]
    rows = []
    for name, r in reports.items():
        dom = r.per_domain()
        cal = r.calibration()
        rows.append(
            "| {} | {:.3f} | {:.1f} | {:.3f} | ".format(
                name, r.accuracy, r.mean_samples, cal["ece"]
            )
            + " | ".join(f"{dom.get(d, float('nan')):.3f}" for d in DOMAINS)
            + " |"
        )
    return "\n".join(header + sep + rows)


def to_json(reports: Dict[str, BenchmarkReport], path: Optional[str] = None) -> str:
    payload = {k: v.as_dict() for k, v in reports.items()}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
