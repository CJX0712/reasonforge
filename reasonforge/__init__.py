"""
reasonforge — Verifiable System-2 Reasoning Engine
可验证 System-2 推理引擎 · 作者：晨星 (Chenxing)

The premise
-----------
A chain of thought is only as good as its weakest **checkable** claim.
``reasonforge`` makes every reasoning step a machine-checkable assertion, routes
each through a family of symbolic verifiers, and uses the verification outcome —
not a popularity contest — to decide the answer.

Layers
------
    L0   numeric primitives          core/numeric.py
    L1   reasoning-trace structures  core/trace.py
    L2   sampling backends           llm/
    L3   search/strategy contract    search/base.py
    L4   verification layer          verifiers/      (arith · algebra · logic · csp · code · consistency)
    L5   aggregation                 aggregate/vwsc.py   ★ VWSC
    L6   inference-time strategies   search/         (SC · BoN · DVTS · UCB bandit)
    L7   adaptive budgeting          core/budget.py
    L8   refinement & calibration    refine/cegar.py, aggregate/calibration.py
    L9   benchmark                   bench/
    L10  orchestration & audit       engine.py

Quick start
-----------
    from reasonforge import ReasonForge
    from reasonforge.bench.tasks import get_task

    rf = ReasonForge(seed=42)
    out = rf.solve(get_task("arith-01"), strategy="vwsc_adaptive", use_cegar=True)
    print(out.answer, out.confidence, out.correct)
    print(rf.explain(out))

Zero hard dependencies: the core runs on the Python standard library alone.
``sympy`` improves algebra verification when present; nothing requires it.
"""

from __future__ import annotations

from .aggregate import (
    PlattCalibrator,
    VWSCConfig,
    calibration_error,
    majority_vote,
    vwsc_aggregate,
)
from .bench.tasks import DOMAINS, TASKS, Task, by_domain, get_task, summary
from .core import (
    Claim,
    DeterministicRNG,
    Step,
    Trace,
    normalize_answer,
)
from .engine import ReasonForge, SolveResult
from .llm import Feedback, GenerationConfig, LLMBackend, MockStudent
from .refine import CegarConfig, build_feedback, cegar_refine
from .search import STRATEGIES, make_strategy
from .verifiers import VerificationReport, VerificationResult, default_registry

__version__ = "1.0.0"
__author__ = "晨星 (Chenxing)"

__all__ = [
    # engine
    "ReasonForge",
    "SolveResult",
    # structures
    "Claim",
    "Step",
    "Trace",
    "normalize_answer",
    "DeterministicRNG",
    # verification
    "default_registry",
    "VerificationReport",
    "VerificationResult",
    # aggregation
    "VWSCConfig",
    "vwsc_aggregate",
    "majority_vote",
    "PlattCalibrator",
    "calibration_error",
    # search
    "STRATEGIES",
    "make_strategy",
    # refinement
    "CegarConfig",
    "cegar_refine",
    "build_feedback",
    # backends
    "MockStudent",
    "LLMBackend",
    "GenerationConfig",
    "Feedback",
    # tasks
    "TASKS",
    "Task",
    "get_task",
    "by_domain",
    "summary",
    "DOMAINS",
    "__version__",
    "__author__",
]
