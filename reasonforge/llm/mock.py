"""
Deterministic mock backend (确定性模拟后端).

Why this exists
---------------
A real LLM is stochastic, costs money, and cannot be unit-tested. This backend
replays the **gold chain** with controlled, seeded corruptions, which lets us
measure — exactly and reproducibly — how much accuracy each inference-time
strategy recovers. It is a *simulated student*, not a language model, and the
docs say so loudly.

Corruption taxonomy (each maps to a distinct failure mode in real LLMs)
-----------------------------------------------------------------------
``clean``        — faithful chain
``slip``         — one intermediate step is arithmetically wrong; the error is
                   propagated to downstream steps so the chain stays *internally
                   consistent* (exactly how LLMs make plausible-but-wrong chains)
``drift``        — every step is right, but the emitted final answer differs
                   (caught only by the consistency verifier)
``truncate``     — final step dropped; answer is an intermediate value
``flip``         — conclusion inverted (logic domain)
``bug``          — code executes but returns a wrong value

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..bench.tasks import Task
from ..core.numeric import DeterministicRNG
from ..core.trace import Claim, Step, Trace
from .base import BackendError, Feedback, GenerationConfig, LLMBackend

__all__ = ["MockStudent", "CORRUPTIONS"]

CORRUPTIONS = ("clean", "slip", "drift", "truncate", "flip", "bug")

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_EQ_SPLIT = re.compile(r"^(?P<lhs>.*?[^=!<>])=(?P<rhs>[^=].*)$")
_DELTAS = (1, 2, -1, -2, 3, -3, 10, -10, 20, -20)


@dataclass
class MockStudent(LLMBackend):
    """Seeded simulated student. Bit-identical across runs and platforms."""

    error_rate: float = 0.45
    weights: Tuple[float, ...] = (0.0, 0.45, 0.20, 0.10, 0.15, 0.10)
    name: str = "mock-student"
    repair_gain: float = 0.55      # how much one counterexample helps
    base_repair: float = 0.20      # repair probability with a single hint
    attractor_prob: float = 0.65   # how often a wrong answer collapses onto a distractor

    def __post_init__(self) -> None:
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError("error_rate must be in [0,1]")
        if not 0.0 <= self.base_repair <= 1.0:
            raise ValueError("base_repair must be in [0,1]")
        if len(self.weights) != len(CORRUPTIONS):
            raise ValueError("weights must align with CORRUPTIONS")
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1")

    # ------------------------------------------------------------------
    def sample(self, task: Task, config: GenerationConfig, rng: DeterministicRNG) -> List[Trace]:
        if not task.gold_claims:
            raise BackendError(f"task {task.id!r} has no gold chain")
        out: List[Trace] = []
        for i in range(config.n):
            stream = rng.fork(i * 7919 + 13)
            kind = self._pick_corruption(stream)
            out.append(self._build_trace(task, i, kind, stream, config))
        return out

    # ------------------------------------------------------------------
    # counterexample-conditioned resampling
    # ------------------------------------------------------------------
    def supports_feedback(self) -> bool:
        return True

    def sample_with_feedback(
        self,
        task: Task,
        config: GenerationConfig,
        rng: DeterministicRNG,
        feedback: Sequence[Feedback],
    ) -> List[Trace]:
        """More (and more specific) counterexamples -> higher repair probability.

        Models the empirically observed behaviour of real LLMs: vague "try again"
        barely helps, while pointing at the exact wrong step with its correct
        value repairs the chain most of the time.
        """
        n_fb = len([f for f in (feedback or []) if f.detail or f.counterexample])
        specific = len([f for f in (feedback or []) if f.suggested])
        gain = 1.0 - math.exp(-(self.repair_gain * n_fb + 0.35 * specific))
        p_repair = min(0.97, self.base_repair + (1.0 - self.base_repair) * gain)
        out: List[Trace] = []
        for i in range(config.n):
            stream = rng.fork(i * 7919 + 13 + 555)
            if stream.random() < p_repair:
                out.append(self._clean(task, i))
            else:
                kind = self._pick_corruption(stream)
                out.append(self._build_trace(task, i, kind, stream, config))
        return out

    # ------------------------------------------------------------------
    def _pick_corruption(self, rng: DeterministicRNG) -> str:
        if rng.random() >= self.error_rate:
            return "clean"
        r = rng.random()
        acc = 0.0
        for kind, w in zip(CORRUPTIONS, self.weights):
            acc += w
            if r < acc:
                return kind
        return "slip"

    def _build_trace(
        self,
        task: Task,
        idx: int,
        kind: str,
        rng: DeterministicRNG,
        config: GenerationConfig,
    ) -> Trace:
        symbolic = task.domain in ("logic", "constraint", "code")
        if kind == "clean":
            built = self._clean(task, idx)
        elif kind == "drift":
            built = self._drift(task, idx, rng)
        elif kind == "truncate":
            built = (
                self._truncate(task, idx, rng)
                if len(task.gold_claims) >= 2
                else self._clean(task, idx)
            )
        elif kind == "flip":
            built = self._semantic_perturb(task, idx, "flip", rng)
        elif kind in ("slip", "bug"):
            built = (
                self._semantic_perturb(task, idx, kind, rng)
                if symbolic
                else self._propagate_slip(task, idx, rng)
            )
        else:
            built = self._clean(task, idx)
        if built.meta.get("corruption") != "clean":
            built = self._apply_attractor(built, task, rng)
        return built

    def _apply_attractor(self, t: Trace, task: Task, rng: DeterministicRNG) -> Trace:
        """Collapse the emitted answer onto a distractor.

        This is the modelling detail that makes majority vote fail the way it
        does in practice: wrong answers are *not* uniformly spread, they pile up
        on a few plausible values. The chain's internal claims are left alone,
        so the consistency verifier sees a contradiction — unless the chain was
        already heading for that value.
        """
        if not task.distractors:
            return t
        if rng.random() >= self.attractor_prob:
            return t
        d = task.distractors[rng.randbelow(len(task.distractors))]
        t.answer = d
        last = t.steps[-1]
        oc = last.claims[0]
        nc = Claim(
            id=oc.id,
            text=f"最终答案为 {d}",
            kind=oc.kind,
            expr=oc.expr,
            deps=oc.deps,
            confidence=oc.confidence,
        )
        t.steps[-1] = Step(id=last.id, text=f"最终答案为 {d}", claims=[nc], score=last.score)
        t.meta["attractor"] = d
        return t

    # ------------------------------------------------------------------
    # builders
    # ------------------------------------------------------------------
    def _clean(self, task: Task, idx: int) -> Trace:
        claims: List[Claim] = []
        steps: List[Step] = []
        for j, (cid, text, expr, k, deps) in enumerate(task.gold_claims):
            claims.append(Claim(id=cid, text=text, kind=k, expr=expr, deps=deps))
            steps.append(Step(id=f"s{idx}_{j}", text=text, claims=[claims[-1]]))
        answer = task.gold_answer
        last_id = task.gold_claims[-1][0]
        claims.append(
            Claim(
                id=f"final_{idx}",
                text=f"最终答案为 {answer}",
                kind="consistency",
                expr=str(answer),
                deps=(last_id,),
            )
        )
        steps.append(Step(id=f"s{idx}_final", text=f"最终答案为 {answer}", claims=[claims[-1]]))
        return Trace(
            id=f"{task.id}#t{idx}",
            steps=steps,
            answer=str(answer),
            answer_claim=f"final_{idx}",
            logprob=-0.15,
            meta={"corruption": "clean", "task": task.id, "domain": task.domain},
        )

    def _drift(self, task: Task, idx: int, rng: DeterministicRNG) -> Trace:
        """Correct steps, wrong emitted answer."""
        t = self._clean(task, idx)
        delta = _DELTAS[rng.randbelow(len(_DELTAS))]
        wrong = _shift_number(str(task.gold_answer), delta)
        t.answer = wrong
        last = t.steps[-1]
        old_claim = last.claims[0]
        # Claim is frozen -> rebuild it. `expr` intentionally keeps the gold
        # value so the consistency verifier can catch the drifted header.
        new_claim = Claim(
            id=old_claim.id,
            text=f"最终答案为 {wrong}",
            kind=old_claim.kind,
            expr=old_claim.expr,
            deps=old_claim.deps,
            confidence=old_claim.confidence,
        )
        t.steps[-1] = Step(id=last.id, text=f"最终答案为 {wrong}", claims=[new_claim], score=last.score)
        t.meta["corruption"] = "drift"
        t.meta["delta"] = delta
        return t

    def _truncate(self, task: Task, idx: int, rng: DeterministicRNG) -> Trace:
        """Drop the final reasoning step; answer becomes an intermediate value."""
        keep = task.gold_claims[:-1]
        claims: List[Claim] = []
        steps: List[Step] = []
        for j, (cid, text, expr, k, deps) in enumerate(keep):
            deps = tuple(d for d in deps if d != task.gold_claims[-1][0])
            claims.append(Claim(id=cid, text=text, kind=k, expr=expr, deps=deps))
            steps.append(Step(id=f"s{idx}_{j}", text=text, claims=[claims[-1]]))
        interim = _last_number(keep[-1][2]) if keep else task.gold_answer
        answer = str(interim) if interim is not None else str(task.gold_answer)
        claims.append(
            Claim(
                id=f"final_{idx}",
                text=f"最终答案为 {answer}",
                kind="consistency",
                expr=answer,
                deps=(keep[-1][0],) if keep else (),
            )
        )
        steps.append(Step(id=f"s{idx}_final", text=f"最终答案为 {answer}", claims=[claims[-1]]))
        return Trace(
            id=f"{task.id}#t{idx}",
            steps=steps,
            answer=answer,
            answer_claim=f"final_{idx}",
            logprob=-0.9,
            meta={"corruption": "truncate", "task": task.id, "domain": task.domain},
        )

    def _propagate_slip(self, task: Task, idx: int, rng: DeterministicRNG) -> Trace:
        """Inject one wrong step, then let the error **propagate correctly**.

        This is the failure mode that makes LLM errors hard to spot: step ``i``
        is false, but every later step is a *valid* consequence of the wrong
        premise — so only a claim-level verifier (or dependency-aware scoring)
        can locate the fault. Downstream values are recomputed by evaluating the
        rewritten left-hand side, so the chain stays internally consistent.
        """
        n = len(task.gold_claims)
        if n == 0:
            return self._clean(task, idx)
        from ..verifiers.arithmetic import safe_eval_arithmetic, _split_equality

        gold_vals = [_last_number(c[2]) for c in task.gold_claims]
        i = rng.randbelow(n)
        delta = _DELTAS[rng.randbelow(len(_DELTAS))]
        if gold_vals[i] is None:
            return self._semantic_perturb(task, idx, "slip", rng)
        new_vals: List[Optional[float]] = list(gold_vals)
        new_vals[i] = float(gold_vals[i]) + delta
        if abs(new_vals[i] - gold_vals[i]) < 1e-12:
            new_vals[i] = float(gold_vals[i]) + 1.0

        exprs: List[str] = []
        for j, (cid, _text, expr, _k, _deps) in enumerate(task.gold_claims):
            if j < i:
                exprs.append(expr)
                continue
            if j == i:
                exprs.append(_replace_last_number(expr, float(new_vals[j])))
                continue
            # downstream: substitute wrong inputs, then recompute the output
            ex = expr
            for k in range(i, j):
                old_v, new_v = gold_vals[k], new_vals[k]
                if old_v is None or new_v is None:
                    continue
                if abs(old_v - new_v) < 1e-12:
                    continue
                ex = _replace_token(ex, _fmt(old_v), _fmt(new_v))
            lhs, rhs = _split_equality(ex)
            computed = safe_eval_arithmetic(lhs) if rhs is not None else None
            if computed is None:
                computed = safe_eval_arithmetic(ex)
            if computed is None or isinstance(computed, bool):
                computed = new_vals[j - 1] if new_vals[j - 1] is not None else gold_vals[j]
            new_vals[j] = float(computed)
            exprs.append(f"{lhs} = {_fmt(float(computed))}" if rhs is not None else ex)

        claims: List[Claim] = []
        steps: List[Step] = []
        for j, (cid, text, _expr, k, deps) in enumerate(task.gold_claims):
            claims.append(Claim(id=cid, text=text, kind=k, expr=exprs[j], deps=deps))
            steps.append(Step(id=f"s{idx}_{j}", text=text, claims=[claims[-1]]))
        final_val = new_vals[-1]
        answer = _fmt(float(final_val)) if final_val is not None else str(task.gold_answer)
        claims.append(
            Claim(
                id=f"final_{idx}",
                text=f"最终答案为 {answer}",
                kind="consistency",
                expr=answer,
                deps=(task.gold_claims[-1][0],),
            )
        )
        steps.append(Step(id=f"s{idx}_final", text=f"最终答案为 {answer}", claims=[claims[-1]]))
        return Trace(
            id=f"{task.id}#t{idx}",
            steps=steps,
            answer=answer,
            answer_claim=f"final_{idx}",
            logprob=-0.6,
            meta={
                "corruption": "slip",
                "perturbed_step": i,
                "delta": delta,
                "task": task.id,
                "domain": task.domain,
            },
        )

    def _semantic_perturb(self, task: Task, idx: int, kind: str, rng: DeterministicRNG) -> Trace:
        """Domain-aware corruption: logic flips, constraint mis-assignments, code bugs."""
        claims: List[Claim] = []
        steps: List[Step] = []
        perturbations: Dict[int, str] = {}
        for j, (cid, text, expr, k, deps) in enumerate(task.gold_claims):
            ex = self._perturb_expr(expr, task.domain, rng)
            if ex != expr:
                perturbations[j] = ex
            claims.append(Claim(id=cid, text=text, kind=k, expr=ex, deps=deps))
            steps.append(Step(id=f"s{idx}_{j}", text=text, claims=[claims[-1]]))
        answer = self._perturb_answer(str(task.gold_answer), task.domain, rng) if perturbations else str(task.gold_answer)
        claims.append(
            Claim(
                id=f"final_{idx}",
                text=f"最终答案为 {answer}",
                kind="consistency",
                expr=answer,
                deps=(task.gold_claims[-1][0],),
            )
        )
        steps.append(Step(id=f"s{idx}_final", text=f"最终答案为 {answer}", claims=[claims[-1]]))
        return Trace(
            id=f"{task.id}#t{idx}",
            steps=steps,
            answer=answer,
            answer_claim=f"final_{idx}",
            logprob=-0.7,
            meta={
                "corruption": kind if perturbations else "clean",
                "perturbed": bool(perturbations),
                "task": task.id,
                "domain": task.domain,
            },
        )

    # ------------------------------------------------------------------
    def _perturb_expr(self, expr: str, domain: str, rng: DeterministicRNG) -> str:
        if domain == "logic":
            return _perturb_logic(expr, rng)
        if domain == "constraint":
            return _perturb_constraint(expr, rng)
        if domain == "code":
            return _perturb_code(expr, rng)
        if domain == "arithmetic":
            delta = _DELTAS[rng.randbelow(len(_DELTAS))]
            return _shift_last_number(expr, delta)
        return expr

    @staticmethod
    def _perturb_answer(answer: str, domain: str, rng: DeterministicRNG) -> str:
        if domain in ("logic",):
            low = answer.strip().lower()
            return "no" if low in ("yes", "true", "1") else "yes"
        if domain in ("arithmetic", "constraint", "code"):
            nums = _NUM_RE.findall(answer)
            if nums:
                delta = _DELTAS[rng.randbelow(len(_DELTAS))]
                return answer.replace(nums[-1], _fmt(float(nums[-1]) + delta))
        return answer


# --------------------------------------------------------------------------
# perturbation helpers
# --------------------------------------------------------------------------
def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.10g}"


def _last_number(expr: str) -> Optional[float]:
    nums = _NUM_RE.findall(expr or "")
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _replace_last_number(expr: str, new_val: float) -> str:
    ms = list(_NUM_RE.finditer(expr))
    if not ms:
        return expr
    m = ms[-1]
    return expr[: m.start()] + _fmt(new_val) + expr[m.end() :]


def _shift_last_number(expr: str, delta: float) -> str:
    """Add ``delta`` to the right-most number of ``expr``."""
    ms = list(_NUM_RE.finditer(expr))
    if not ms:
        return expr
    m = ms[-1]
    try:
        new = _fmt(float(m.group(0)) + delta)
    except ValueError:
        return expr
    return expr[: m.start()] + new + expr[m.end() :]


def _replace_token(expr: str, old: str, new: str) -> str:
    return re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d.])", new, expr)


def _shift_number(text: str, delta: float) -> str:
    nums = _NUM_RE.findall(text)
    if not nums:
        return text + f" {int(delta)}"
    return text.replace(nums[-1], _fmt(float(nums[-1]) + delta))


def _perturb_logic(expr: str, rng: DeterministicRNG) -> str:
    s = expr.strip()
    m = re.match(r"^\s*(SAT|UNSAT|VALID|INVALID)\s*[:：]\s*(?P<b>.*)$", s, re.IGNORECASE)
    if m:
        mode = m.group(1).upper()
        body = m.group("b")
        flip = {"SAT": "UNSAT", "UNSAT": "SAT", "VALID": "INVALID", "INVALID": "VALID"}
        return f"{flip[mode]}: {body}"
    if "|=" in s:
        left, right = s.split("|=", 1)
        premises = [p.strip() for p in re.split(r"[;,\n]+", left) if p.strip()]
        if len(premises) >= 2 and rng.random() < 0.5:
            premises[0], premises[-1] = premises[-1], premises[0]
            return ", ".join(premises) + " |= " + right.strip()
        return right.strip() + " |= " + ", ".join(premises)
    return f"SAT: not ({s})"


def _perturb_constraint(expr: str, rng: DeterministicRNG) -> str:
    if "###" not in expr:
        return expr
    left, right = expr.split("###", 1)
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)", left)
    if not m:
        return expr
    delta = _DELTAS[rng.randbelow(len(_DELTAS))]
    new = _fmt(float(m.group(2)) + delta)
    left = left[: m.start(2)] + new + left[m.end(2) :]
    return left + "###" + right


def _perturb_code(expr: str, rng: DeterministicRNG) -> str:
    if "###" not in expr:
        return expr
    code, expected = expr.split("###", 1)
    m = re.search(r"^(?P<indent>[ \t]*)result\s*=\s*(?P<val>.+)$", code, re.MULTILINE)
    delta = _DELTAS[rng.randbelow(len(_DELTAS))]
    if m:
        new_line = f"{m.group('indent')}result = ({m.group('val')}) + ({delta})"
        code = code[: m.start()] + new_line + code[m.end() :]
        return code + "###" + expected
    # fall back to shrinking a range upper bound
    m2 = re.search(r"range\((?P<a>\d+),\s*(?P<b>\d+)\)", code)
    if m2:
        b = max(int(m2.group("a")) + 1, int(m2.group("b")) - 1)
        code = code[: m2.start()] + f"range({m2.group('a')}, {b})" + code[m2.end() :]
    return code + "###" + expected
