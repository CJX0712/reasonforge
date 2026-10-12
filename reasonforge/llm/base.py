"""
L2 — LLM backend contract (模型后端契约).

A backend's only job is to turn a :class:`~reasonforge.bench.tasks.Task` into a
list of candidate :class:`~reasonforge.core.trace.Trace` objects. Everything
downstream (verification, aggregation, refinement) is backend-agnostic.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..bench.tasks import Task
from ..core.numeric import DeterministicRNG
from ..core.trace import Trace

__all__ = ["GenerationConfig", "LLMBackend", "BackendError", "Feedback"]


@dataclass
class Feedback:
    """A concrete, machine-derived correction handed back to the model.

    This is CEGAR's "counterexample" — not a vague "try again", but the exact
    claim that failed and the exact value it should have had.
    """

    claim_id: str = ""
    kind: str = ""
    expr: str = ""
    detail: str = ""
    counterexample: str = ""
    suggested: Optional[str] = None
    trace_id: str = ""

    def render(self) -> str:
        parts = [f"- claim {self.claim_id} [{self.kind}]: {self.expr} is WRONG"]
        if self.counterexample:
            parts.append(f"  evidence: {self.counterexample}")
        if self.suggested:
            parts.append(f"  should be: {self.suggested}")
        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "kind": self.kind,
            "expr": self.expr,
            "detail": self.detail,
            "counterexample": self.counterexample,
            "suggested": self.suggested,
            "trace_id": self.trace_id,
        }


@dataclass
class GenerationConfig:
    """Sampling knobs shared by every backend."""

    n: int = 8
    temperature: float = 0.8
    top_p: float = 0.95
    max_steps: int = 8

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be in (0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


class BackendError(RuntimeError):
    """Raised when a backend cannot fulfil a sampling request."""


class LLMBackend(ABC):
    """Abstract sampling backend."""

    name: str = "abstract"

    @abstractmethod
    def sample(self, task: Task, config: GenerationConfig, rng: DeterministicRNG) -> List[Trace]:
        """Return ``config.n`` candidate traces for ``task``."""

    def supports_feedback(self) -> bool:
        """Whether :meth:`sample_with_feedback` does more than plain sampling."""
        return False

    def sample_with_feedback(
        self,
        task: Task,
        config: GenerationConfig,
        rng: DeterministicRNG,
        feedback: Sequence["Feedback"],
    ) -> List[Trace]:
        """Re-sample while conditioning on concrete counterexamples.

        Backends that cannot condition on feedback simply fall back to
        :meth:`sample` — correctness never depends on this capability.
        """
        return self.sample(task, config, rng)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"
