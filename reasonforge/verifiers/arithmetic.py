"""
Arithmetic verifier (算术验证器).

Whitelisted-AST evaluation — **never** ``eval``. Supports equality assertions
(``"12 * 5 = 60"``), bare evaluations (``"12 * 5"``), percentages and a small
constant/function table.

Hardening
---------
* ``Pow`` exponent is capped (``|exp| <= 64`` and ``|base| <= 1e6``) to stop
  accidental ``9**9**9`` blow-ups.
* Division by zero yields ``None`` (unknown), never a crash.
* Float comparison uses a relative+absolute tolerance (1e-9 / 1e-12).

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, Optional

from ..core.trace import Claim, Trace
from .base import Verifier, VerificationResult

__all__ = ["ArithmeticVerifier", "safe_eval_arithmetic", "ArithError"]

_REL_TOL = 1e-9
_ABS_TOL = 1e-12

_CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}

_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": lambda xs, *a: sum(xs, *a),
    "len": len,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "int": int,
    "float": float,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Tuple,
    ast.List,
    ast.IfExp,
    ast.keyword,
)

_PCT_RE = re.compile(r"(?P<a>[-+]?\d+(?:\.\d+)?)\s*%\s*(?:of\s*)?(?P<b>[-+]?\d+(?:\.\d+)?)")


class ArithError(Exception):
    """Raised for expressions outside the supported arithmetic subset."""


def _check(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_NODES):
            raise ArithError(f"disallowed syntax: {type(n).__name__}")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _FUNCTIONS:
                raise ArithError("only whitelisted functions may be called")


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool)):
            raise ArithError("only numeric literals allowed")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ArithError(f"unknown name {node.id!r}")
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left), _eval(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            if b == 0:
                raise ArithError("division by zero")
            return a / b
        if isinstance(op, ast.FloorDiv):
            if b == 0:
                raise ArithError("division by zero")
            return a // b
        if isinstance(op, ast.Mod):
            if b == 0:
                raise ArithError("modulo by zero")
            return a % b
        if isinstance(op, ast.Pow):
            if abs(a) > 1e6 or abs(b) > 64:
                raise ArithError("power operands out of safe range")
            return a**b
        raise ArithError("unsupported operator")
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.Compare):
        left = _eval(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp)
            if isinstance(op, ast.Eq):
                ok = _close(left, right)
            elif isinstance(op, ast.NotEq):
                ok = not _close(left, right)
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right or _close(left, right)
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right or _close(left, right)
            else:
                raise ArithError("unsupported comparison")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Not):
        return not _eval(node.operand)
    if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.IfExp):
        return _eval(node.body) if _eval(node.test) else _eval(node.orelse)
    if isinstance(node, ast.Call):
        fn = _FUNCTIONS[node.func.id]  # type: ignore[union-attr]
        args = [_eval(a) for a in node.args]
        return fn(*args)
    raise ArithError(f"unsupported node {type(node).__name__}")


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return False
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= max(_ABS_TOL, _REL_TOL * max(abs(a), abs(b)))
    return a == b


def _eval_single(src: str) -> Optional[Any]:
    """Evaluate one side of an expression (no top-level equality allowed)."""
    # NB: leading whitespace makes ast.parse raise IndentationError, not
    # SyntaxError — strip first, or "= 10" silently evaluates to None.
    src = src.strip()
    if not src:
        return None
    # percentage sugar: "15% of 200" -> "0.15 * 200"
    if "%" in src:
        src = _PCT_RE.sub(lambda m: f"({m.group('a')}/100 * {m.group('b')})", src)
    src = src.replace("^", "**")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        return None
    try:
        _check(tree)
        return _eval(tree)
    except (ArithError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def safe_eval_arithmetic(expr: str) -> Optional[Any]:
    """Evaluate ``expr`` inside the arithmetic subset.

    A bare expression returns its value; an assertion ``"<lhs> = <rhs>"``
    returns the boolean result of comparing both sides (because a single ``=``
    is not valid Python, it must be split before parsing). ``None`` if anything
    is unsupported or malformed.
    """
    if expr is None:
        return None
    src = str(expr).strip()
    if not src:
        return None
    lhs, rhs = _split_equality(src)
    if rhs is None:
        return _eval_single(src)
    lv = _eval_single(lhs)
    rv = _eval_single(rhs)
    if lv is None or rv is None:
        return None
    return _close(lv, rv)


class ArithmeticVerifier(Verifier):
    """Verifies ``kind="arithmetic"`` claims.

    Accepts ``"<lhs> = <rhs>"`` / ``"<lhs> == <rhs>"`` assertions; a bare
    expression is reported as **unknown** with its computed value in ``detail``
    (we cannot call it correct without something to compare against).
    """

    kind = "arithmetic"
    name = "arithmetic-verifier"

    def verify(self, claim: Claim, trace: Optional[Trace] = None) -> VerificationResult:
        if not self.supports(claim) or claim.expr is None:
            return self._unsupported(claim)
        expr = claim.expr.strip()
        # Split on a single top-level '=' that is NOT '==', '<=', '>=', '!='
        lhs, rhs = _split_equality(expr)
        if rhs is None:
            val = safe_eval_arithmetic(expr)
            if val is None:
                return self._result(claim, None, 0.0, f"unparseable expression: {expr!r}")
            return self._result(claim, None, 0.3, f"evaluates to {val!r} (no assertion to check)")
        lv = safe_eval_arithmetic(lhs)
        rv = safe_eval_arithmetic(rhs)
        if lv is None or rv is None:
            bad = lhs if lv is None else rhs
            return self._result(claim, None, 0.0, f"unparseable side: {bad!r}")
        ok = _close(lv, rv)
        if ok:
            return self._result(claim, True, 1.0, f"both sides = {lv!r}")
        return self._result(
            claim,
            False,
            1.0,
            f"LHS={lv!r} != RHS={rv!r}",
            counterexample=f"{lhs.strip()} = {lv!r}, not {rv!r}",
        )


def _split_equality(expr: str) -> tuple:
    """Return ``(lhs, rhs)`` or ``(expr, None)`` if there is no assertion."""
    depth = 0
    for i in range(len(expr) - 1):
        ch = expr[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            nxt = expr[i + 1]
            prv = expr[i - 1] if i > 0 else ""
            if nxt == "=" or prv == "=" or prv in "!<>":
                continue
            return expr[:i], expr[i + 1 :]
    return expr, None


def arithmetic_ok(expr: str) -> Optional[bool]:
    """Convenience: ``True/False/None`` for a standalone arithmetic assertion."""
    v = ArithmeticVerifier()
    c = Claim(id="tmp", text=expr, kind="arithmetic", expr=expr)
    r: VerificationResult = v.verify(c)
    return r.ok
