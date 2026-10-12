"""
L8 — CEGAR: Counterexample-Guided Abstraction Refinement for reasoning chains.

Borrowed from formal methods (Clarke et al.), where a failed verification returns
a *counterexample* that refines the abstraction. Here the abstraction is the
chain of thought: when the verifier refutes a claim, we hand the model the exact
step, the exact wrong value and — when we can compute it — the exact right value,
then re-sample.

Crucially this is **not** "try again". Vague retry barely moves accuracy; a
concrete counterexample repairs the chain with high probability. The
:class:`Feedback` payload is what makes CEGAR work, and the verifier is what
produces it — so verification and refinement form a closed loop.

Loop
----
1. sample a batch
2. verify every claim
3. aggregate with VWSC
4. if decisive -> stop; else build feedback and re-sample (bounded rounds)

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.trace import Trace
from ..llm.base import Feedback, GenerationConfig
from ..verifiers.arithmetic import _split_equality, safe_eval_arithmetic
from ..verifiers.base import VerificationReport, VerificationResult
from ..verifiers.constraint import CSP, solve_csp

__all__ = [
    "CegarConfig",
    "build_feedback",
    "suggest_fix",
    "cegar_refine",
    "CegarRefinement",
]


@dataclass
class CegarConfig:
    max_rounds: int = 3
    target_confidence: float = 0.70
    min_trust: float = 0.60
    max_feedback: int = 5
    resample_per_round: int = 6
    require_ungated: int = 2

    def __post_init__(self) -> None:
        if self.max_rounds < 0:
            raise ValueError("max_rounds >= 0")
        if self.resample_per_round < 1:
            raise ValueError("resample_per_round >= 1")


# --------------------------------------------------------------------------
# feedback construction
# --------------------------------------------------------------------------
def suggest_fix(claim, result: VerificationResult) -> Optional[str]:
    """Best-effort repair hint: what the claim *should* have said.

    Returns ``None`` when no machine-derivable correction exists (then the
    feedback is a refutation without a suggestion — still useful, weaker).
    """
    if not claim.expr:
        return None
    kind = claim.kind
    try:
        if kind == "arithmetic":
            lhs, rhs = _split_equality(claim.expr)
            if rhs is None:
                return None
            val = safe_eval_arithmetic(lhs)
            if val is None:
                return None
            if isinstance(val, float) and abs(val - round(val)) < 1e-9:
                val = int(round(val))
            return f"{lhs.strip()} = {val}"
        if kind == "constraint":
            if "###" not in claim.expr:
                return None
            _left, right = claim.expr.split("###", 1)
            cons = [c.strip() for c in re.split(r"[,;\n]+", right) if c.strip()]
            names: List[str] = []
            for c in cons:
                for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", c):
                    if m.group(1) not in names:
                        names.append(m.group(1))
            if not names:
                return None
            dom = {n: [float(i) for i in range(-20, 21)] for n in names}
            model = solve_csp(CSP(dom, cons))
            if not model:
                return None
            return "; ".join(
                f"{k}={int(v) if float(v).is_integer() else v}" for k, v in model.items()
            ) + " ### " + ", ".join(cons)
        if kind == "consistency" and result.counterexample:
            m = re.search(r"claim asserts\s+(.+)$", result.counterexample)
            if m:
                return m.group(1).strip()
            return None
    except Exception:
        return None
    return None


def build_feedback(
    traces: Sequence[Trace],
    reports: Dict[str, VerificationReport],
    limit: int = 5,
) -> List[Feedback]:
    """Collect counterexamples from all refuted claims.

    Ordered by "actionability": claims with a derivable fix first, then deeper
    claims (deeper failures are more likely to be the root cause than leaves).
    """
    items: List[Tuple[int, int, Feedback]] = []
    for t in traces:
        rep = reports.get(t.id)
        if rep is None:
            continue
        try:
            depth_map = {c.id: i for i, c in enumerate(t.topo_order())}
        except Exception:
            depth_map = {c.id: i for i, c in enumerate(t.claims())}
        cmap = {c.id: c for c in t.claims()}
        for cid, res in rep.results.items():
            if not res.failed:
                continue
            claim = cmap.get(cid)
            if claim is None:
                continue
            fix = suggest_fix(claim, res)
            fb = Feedback(
                claim_id=cid,
                kind=claim.kind,
                expr=claim.expr or "",
                detail=res.detail,
                counterexample=res.counterexample or "",
                suggested=fix,
                trace_id=t.id,
            )
            has_fix = 0 if fix else 1
            items.append((has_fix, -depth_map.get(cid, 0), fb))
    items.sort(key=lambda x: (x[0], x[1], x[2].claim_id))
    dedup: List[Feedback] = []
    seen = set()
    for _a, _b, fb in items:
        key = (fb.kind, fb.expr, fb.suggested)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(fb)
        if len(dedup) >= limit:
            break
    return dedup


# --------------------------------------------------------------------------
# the refinement loop
# --------------------------------------------------------------------------
@dataclass
class CegarRefinement:
    """Result of a CEGAR run."""

    traces: List[Trace] = field(default_factory=list)
    rounds: List[Dict[str, object]] = field(default_factory=list)
    feedback_used: List[List[Feedback]] = field(default_factory=list)
    n_rounds: int = 0
    converged: bool = False
    stop_reason: str = "exhausted"

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_rounds": self.n_rounds,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "rounds": self.rounds,
            "n_traces": len(self.traces),
            "feedback": [[f.as_dict() for f in r] for r in self.feedback_used],
        }


def cegar_refine(
    ctx,
    initial: List[Trace],
    cfg: Optional[CegarConfig] = None,
) -> CegarRefinement:
    """Run the counterexample-guided refinement loop.

    ``ctx`` is a :class:`~reasonforge.search.base.SolveContext`.
    """
    from ..aggregate.vwsc import vwsc_aggregate
    from ..search.base import verify_all as _verify_all

    cfg = cfg or CegarConfig()
    traces: List[Trace] = list(initial)
    ref = CegarRefinement(traces=traces)

    agg: Dict[str, object] = {}
    for r in range(cfg.max_rounds + 1):
        reports = _verify_all(traces, ctx.registry)
        agg = vwsc_aggregate([t for t in traces if t.id in reports], reports, ctx.vwsc)
        conf = float(agg.get("confidence", 0.0))
        trust = float(agg.get("mean_trust", 0.0))
        kept = int(agg.get("n_kept", 0))
        ref.rounds.append(
            {
                "round": r,
                "n_traces": len(traces),
                "confidence": conf,
                "mean_trust": trust,
                "n_kept": kept,
                "n_gated": int(agg.get("n_gated", 0)),
                "answer": str(agg.get("answer", "")),
            }
        )
        if conf >= cfg.target_confidence and trust >= cfg.min_trust and kept >= cfg.require_ungated:
            ref.converged = True
            ref.stop_reason = "converged"
            break
        if r == cfg.max_rounds:
            ref.stop_reason = "max_rounds"
            break

        feedback = build_feedback(traces, reports, cfg.max_feedback)
        ref.feedback_used.append(feedback)
        gen = GenerationConfig(
            n=cfg.resample_per_round,
            temperature=ctx.gen.temperature,
            top_p=ctx.gen.top_p,
        )
        new_traces = ctx.backend.sample_with_feedback(
            ctx.task, gen, ctx.rng.fork(900 + r), feedback
        )
        # re-id to keep trace identity unique across rounds
        for i, t in enumerate(new_traces):
            t.id = f"{t.id}@r{r+1}_{i}"
            t.meta["cegar_round"] = r + 1
        traces.extend(new_traces)
        ref.traces = traces

    ref.n_rounds = len(ref.rounds) - 1
    ref.traces = traces
    return ref
