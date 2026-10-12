"""
Algebra verifier (代数验证器) — sympy first, deterministic fallback second.

Zero-rewrite policy: when ``sympy`` is installed we delegate to the world-class
symbolic engine (exact simplification, exact equation solving). When it is not,
we fall back to a **sampling identity test**: evaluate both sides at many
deterministic points; disagreement proves inequality, agreement across N points
gives high (but not unit) confidence.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.numeric import DeterministicRNG
from ..core.trace import Claim, Trace
from .arithmetic import _split_equality
from .base import Verifier, VerificationResult

__all__ = ["AlgebraVerifier", "HAS_SYMPY", "sampling_identity_test"]

try:  # pragma: no cover - environment dependent
    import sympy as _sympy

    HAS_SYMPY = True
except Exception:  # pragma: no cover
    _sympy = None  # type: ignore[assignment]
    HAS_SYMPY = False


_SAFE_SRC_CHARS = re.compile(r"^[0-9A-Za-z_ \t\+\-\*/\^\(\)\.,=<>!&\|\s]*$")
_N_SAMPLES = 24


def _sympify(expr: str):
    """Parse with ``sympy.parsing.sympy_parser`` using a restricted transform."""
    if not HAS_SYMPY:
        return None
    src = expr.strip().replace("^", "**")
    if not src or not _SAFE_SRC_CHARS.match(src):
        return None
    try:
        return _sympy.sympify(src, rational=True, convert_xor=False)
    except Exception:
        return None


def sampling_identity_test(lhs: str, rhs: str, n: int = _N_SAMPLES, seed: int = 7) -> Tuple[Optional[bool], str]:
    """Fallback identity test by deterministic random sampling.

    Returns ``(ok, detail)`` with ``ok`` in ``{True, False, None}``.
    A single disagreeing sample refutes the identity (``False``, sound).
    """
    from .arithmetic import safe_eval_arithmetic

    names = sorted(_identifier_names(lhs) | _identifier_names(rhs))
    rng = DeterministicRNG(seed)
    agrees = 0
    tested = 0
    for _ in range(n):
        env = {nm: round(rng.gauss(0.0, 3.0), 4) for nm in names}
        a = safe_eval_arithmetic(_subst(lhs, env))
        b = safe_eval_arithmetic(_subst(rhs, env))
        if a is None or b is None:
            continue
        if isinstance(a, bool) or isinstance(b, bool):
            continue
        tested += 1
        if not math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-9):
            return False, f"disagrees at {env}: LHS={a!r} RHS={b!r}"
        agrees += 1
    if tested == 0:
        return None, "expression could not be sampled"
    return (None if tested < 3 else True), f"{agrees}/{tested} sample points agree"


def _identifier_names(expr: str) -> set:
    out = set()
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr):
        nm = m.group(1)
        if nm in ("and", "or", "not", "True", "False"):
            continue
        out.add(nm)
    return out


def _subst(expr: str, env: Dict[str, float]) -> str:
    def rep(m: "re.Match") -> str:
        nm = m.group(0)
        if nm in env:
            v = env[nm]
            return f"({v})" if v < 0 else f"{v}"
        return nm

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rep, expr)


class AlgebraVerifier(Verifier):
    """Verifies ``kind="algebra"`` claims (symbolic identities and equations).

    Accepted ``expr``::

        "(x+1)^2 = x^2 + 2*x + 1"      identity
        "x**2 - 4 = 0"                equation -> checked for real solutions
    """

    kind = "algebra"
    name = "algebra-verifier"

    def verify(self, claim: Claim, trace: Optional[Trace] = None) -> VerificationResult:
        if not self.supports(claim) or claim.expr is None:
            return self._unsupported(claim)
        expr = claim.expr.strip()
        lhs, rhs = _split_equality(expr)

        # --- sympy path: exact -------------------------------------------
        if HAS_SYMPY:
            res = self._sympy_check(claim, lhs, rhs)
            if res is not None:
                return res

        # --- fallback path ------------------------------------------------
        if rhs is None:
            return self._result(claim, None, 0.0, "no equality to check and sympy unavailable")
        ok, detail = sampling_identity_test(lhs, rhs)
        if ok is False:
            return self._result(claim, False, 0.99, detail, counterexample=detail)
        if ok is True:
            return self._result(claim, True, 0.9, f"sampled identity: {detail}")
        return self._result(claim, None, 0.2, detail)

    # ------------------------------------------------------------------
    def _sympy_check(self, claim: Claim, lhs: str, rhs: Optional[str]) -> Optional[VerificationResult]:
        try:
            if rhs is not None:
                a = _sympify(lhs)
                b = _sympify(rhs)
                if a is None or b is None:
                    return None
                diff = _sympy.simplify(a - b)
                if diff == 0:
                    return self._result(claim, True, 1.0, f"exact identity: {lhs} - ({rhs}) = 0")
                # maybe it holds only at specific points -> report as equation
                return self._result(
                    claim,
                    False,
                    1.0,
                    f"not an identity: difference simplifies to {diff}",
                    counterexample=f"difference = {diff}",
                )
            # bare equation: does it have solutions?
            e = _sympify(lhs)
            if e is None:
                return None
            if not isinstance(e, _sympy.Equality) and not hasattr(e, "lhs"):
                return None
            sols = _sympy.solve(e, dict=True)
            if sols:
                return self._result(claim, True, 1.0, f"equation has solutions: {sols}")
            return self._result(claim, None, 0.3, "no symbolic solutions found")
        except Exception:
            return None
