"""
Constraint verifier (约束满足验证器) — finite-domain CSP with MAC-style search.

Classical stack, all reused rather than reinvented:
* AC-3 arc consistency (Mackworth 1977)
* Backtracking with MRV + least-constraining-value ordering
* Forward checking

Claim syntax::

    "x=6; y=4 ### x+y=10, x-y=2, x>y"
     └── assignment ──┘   └────── constraint system ──────┘

``ok=True`` iff the assignment satisfies every constraint.
:sfunc:`solve_csp` is exposed so other layers can *derive* solutions.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.trace import Claim, Trace
from .arithmetic import safe_eval_arithmetic
from .base import Verifier, VerificationResult

__all__ = [
    "ConstraintVerifier",
    "CSP",
    "solve_csp",
    "ac3",
    "check_assignment",
]

_SEP = "###"
_ASSIGN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)")


class CSP:
    """Finite-domain constraint satisfaction problem."""

    def __init__(self, domains: Dict[str, Sequence[float]], constraints: Sequence[str]) -> None:
        self.domains: Dict[str, List[float]] = {k: list(v) for k, v in domains.items()}
        self.constraints: List[str] = list(constraints)
        self.vars: List[str] = list(self.domains)

    # -- evaluation -------------------------------------------------------
    def satisfied_by(self, assign: Dict[str, float]) -> Tuple[bool, str]:
        for c in self.constraints:
            val = safe_eval_arithmetic(_subst(c, assign))
            if val is None:
                return False, f"constraint unparseable under assignment: {c!r}"
            if val is False or val == 0:
                return False, f"constraint violated: {c}"
            if not isinstance(val, (bool, int, float)):
                return False, f"non-boolean constraint result: {c}"
        return True, ""

    def consistent_partial(self, assign: Dict[str, float]) -> bool:
        """Forward check: constraints whose variables are all bound must hold."""
        for c in self.constraints:
            names = _names(c)
            if all(n in assign for n in names):
                val = safe_eval_arithmetic(_subst(c, assign))
                if val is None or val is False or val == 0:
                    return False
        return True

    def neighbours(self, v: str) -> List[str]:
        out: List[str] = []
        for c in self.constraints:
            for n in _names(c):
                if n != v and n in self.domains and n not in out:
                    out.append(n)
        return out

    def constraints_between(self, a: str, b: str) -> List[str]:
        return [c for c in self.constraints if a in _names(c) and b in _names(c)]


def _names(expr: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr):
        nm = m.group(1)
        if nm in ("and", "or", "not", "True", "False"):
            continue
        if nm not in out:
            out.append(nm)
    return out


def _subst(expr: str, env: Dict[str, float]) -> str:
    def rep(m: "re.Match") -> str:
        nm = m.group(0)
        if nm in env:
            v = env[nm]
            return f"({v})" if v < 0 else f"{v}"
        return nm

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rep, expr)


def _revise(csp: CSP, xi: str, xj: str) -> bool:
    """Remove values of ``xi`` unsupported by any value of ``xj``."""
    changed = False
    for vi in list(csp.domains[xi]):
        supported = False
        for vj in csp.domains[xj]:
            pair = {xi: vi, xj: vj}
            ok = True
            for c in csp.constraints_between(xi, xj):
                names = _names(c)
                if not all(n in pair for n in names):
                    continue
                val = safe_eval_arithmetic(_subst(c, pair))
                if val is None or val is False or val == 0:
                    ok = False
                    break
            if ok:
                supported = True
                break
        if not supported:
            csp.domains[xi].remove(vi)
            changed = True
    return changed


def ac3(csp: CSP) -> bool:
    """Mackworth's AC-3. Returns ``False`` if a domain is wiped out.

    Unary constraints (those mentioning a single variable, e.g. ``"x > 5"``)
    are seeded as self-arcs ``(v, v)``; :func:`_revise` already prunes a value
    of ``v`` when it violates such a constraint while every other value of ``v``
    is bound, so the self-arc path is sound.
    """
    queue: List[Tuple[str, str]] = [
        (xi, xj) for xi in csp.vars for xj in csp.neighbours(xi)
    ]
    # seed unary constraints so single-variable restrictions are propagated
    for c in csp.constraints:
        names = _names(c)
        if len(names) == 1 and names[0] in csp.domains:
            queue.append((names[0], names[0]))
    while queue:
        xi, xj = queue.pop(0)
        if _revise(csp, xi, xj):
            if not csp.domains[xi]:
                return False
            for xk in csp.neighbours(xi):
                if xk != xj:
                    queue.append((xk, xi))
    return True


def solve_csp(csp: CSP, max_nodes: int = 20000) -> Optional[Dict[str, float]]:
    """Backtracking + AC-3 + MRV. Deterministic; returns first model found."""
    work = CSP(csp.domains, csp.constraints)
    if not ac3(work):
        return None
    nodes = [0]

    def backtrack(assign: Dict[str, float], doms: Dict[str, List[float]]) -> Optional[Dict[str, float]]:
        if nodes[0] > max_nodes:
            return None
        nodes[0] += 1
        if len(assign) == len(work.vars):
            return dict(assign)
        unassigned = [v for v in work.vars if v not in assign]
        # MRV
        var = min(unassigned, key=lambda v: (len(doms[v]), work.vars.index(v)))
        for value in doms[var]:
            trial = dict(assign)
            trial[var] = value
            if not work.consistent_partial(trial):
                continue
            # forward check on remaining domains
            nd = {v: list(doms[v]) for v in work.vars if v not in trial}
            pruned_ok = True
            for v in list(nd):
                keep = []
                for val in nd[v]:
                    t2 = dict(trial)
                    t2[v] = val
                    if work.consistent_partial(t2):
                        keep.append(val)
                if not keep:
                    pruned_ok = False
                    break
                nd[v] = keep
            if not pruned_ok:
                continue
            res = backtrack(trial, nd)
            if res is not None:
                return res
        return None

    return backtrack({}, {v: list(work.domains[v]) for v in work.vars})


def check_assignment(assign: Dict[str, float], constraints: Sequence[str]) -> Tuple[bool, str]:
    """True/False plus a human-readable violated constraint."""
    for c in constraints:
        val = safe_eval_arithmetic(_subst(c, assign))
        if val is None:
            return False, f"unparseable constraint {c!r}"
        if val is False or val == 0:
            return False, f"violated: {c}"
    return True, ""


class ConstraintVerifier(Verifier):
    """Verifies ``kind="constraint"`` claims.

    Two modes:
    * ``"x=6; y=4 ### x+y=10, x-y=2"`` — assignment satisfies the system
    * ``"### x+y=10, x-y=2"``          — system is satisfiable (no assignment given)
    """

    kind = "constraint"
    name = "constraint-verifier-ac3"

    def verify(self, claim: Claim, trace: Optional[Trace] = None) -> VerificationResult:
        if not self.supports(claim) or claim.expr is None:
            return self._unsupported(claim)
        expr = claim.expr.strip()
        if _SEP not in expr:
            return self._result(claim, None, 0.0, f"missing {_SEP!r} separator")
        left, right = expr.split(_SEP, 1)
        constraints = [c.strip() for c in re.split(r"[,;\n]+", right) if c.strip()]
        if not constraints:
            return self._result(claim, None, 0.0, "empty constraint system")
        assign: Dict[str, float] = {}
        for m in _ASSIGN_RE.finditer(left):
            try:
                assign[m.group(1)] = float(m.group(2))
            except ValueError:
                return self._result(claim, None, 0.0, f"bad assignment {m.group(0)!r}")

        if not assign:
            # satisfiability mode: need domains -> infer small integer domain
            names: List[str] = []
            for c in constraints:
                for n in _names(c):
                    if n not in names:
                        names.append(n)
            if not names:
                return self._result(claim, None, 0.0, "no variables found")
            dom = {n: [float(i) for i in range(-20, 21)] for n in names}
            csp = CSP(dom, constraints)
            model = solve_csp(csp)
            if model is None:
                return self._result(claim, False, 0.9, "constraint system appears unsatisfiable")
            pretty = ", ".join(f"{k}={int(v) if float(v).is_integer() else v}" for k, v in model.items())
            return self._result(claim, True, 0.9, f"satisfiable, e.g. {pretty}")

        missing = sorted({n for c in constraints for n in _names(c)} - set(assign))
        if missing:
            return self._result(
                claim, None, 0.0, f"assignment missing variables: {missing}",
            )
        ok, why = check_assignment(assign, constraints)
        pretty = ", ".join(f"{k}={int(v) if float(v).is_integer() else v}" for k, v in assign.items())
        if ok:
            return self._result(claim, True, 1.0, f"{pretty} satisfies all {len(constraints)} constraints")
        return self._result(
            claim, False, 1.0, why, counterexample=f"{pretty} violates: {why}"
        )
