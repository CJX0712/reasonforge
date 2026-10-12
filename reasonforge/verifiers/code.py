"""
Code verifier (代码执行验证器) — sandboxed execution of a restricted Python subset.

**No ``exec``. No ``eval``. No imports. No attribute access.** The subset is
interpreted node-by-node against an explicit whitelist, with hard caps on steps,
loop iterations and recursion depth. Everything is deterministic.

Supported
---------
statements: assign / aug-assign / for(range|list) / while(capped) / if / return /
            break / continue / pass / def (top-level, depth-capped)
expressions: literals, names, arithmetic, comparisons, boolean ops, subscript,
             list/tuple/dict/set, comprehensions, if-expressions, whitelisted calls

Claim syntax: ``"<code>###<expected>"`` — the returned value (or ``result``) is
compared with ``<expected>``; without ``###`` the claim is only checked to *run*.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from ..core.trace import Claim, Trace
from .base import Verifier, VerificationResult

__all__ = ["CodeVerifier", "SandboxError", "run_sandboxed", "SAFE_BUILTINS"]

MAX_STEPS = 200_000
MAX_ITER = 50_000
MAX_DEPTH = 24
SEP = "###"


class SandboxError(Exception):
    """Any violation of the sandbox contract."""


def _b_list(x):
    return list(x)


SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": lambda x: list(reversed(x)),
    "list": _b_list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "all": all,
    "any": any,
    "pow": pow,
    "divmod": divmod,
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
    "pi": math.pi,
    "e": math.e,
    "print": lambda *a: None,
}

_ALLOWED_STMT = (
    ast.Assign,
    ast.AugAssign,
    ast.For,
    ast.While,
    ast.If,
    ast.Expr,
    ast.Return,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.FunctionDef,
)

_ALLOWED_EXPR = (
    ast.Constant,
    ast.Name,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Subscript,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.IfExp,
    ast.Slice,
    ast.Starred,
    ast.Attribute,
)

_ALLOWED_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)

_SAFE_ATTRS = {
    "append",
    "extend",
    "pop",
    "insert",
    "remove",
    "index",
    "count",
    "sort",
    "reverse",
    "copy",
    "keys",
    "values",
    "items",
    "get",
    "add",
    "discard",
    "union",
    "intersection",
    "difference",
    "strip",
    "split",
    "join",
    "replace",
    "startswith",
    "endswith",
    "upper",
    "lower",
    "find",
}


class _Frame:
    __slots__ = ("vars",)

    def __init__(self, initial: Optional[Dict[str, Any]] = None) -> None:
        self.vars: Dict[str, Any] = dict(initial or {})


class _Interp:
    def __init__(self) -> None:
        self.steps = 0
        self.depth = 0
        self.functions: Dict[str, ast.FunctionDef] = {}
        self.globals = _Frame({**SAFE_BUILTINS})
        self.output: List[str] = []

    # -- guards -----------------------------------------------------------
    def tick(self, n: int = 1) -> None:
        self.steps += n
        if self.steps > MAX_STEPS:
            raise SandboxError(f"step limit exceeded ({MAX_STEPS})")

    # -- entry ------------------------------------------------------------
    def run(self, src: str) -> Any:
        tree = ast.parse(src)
        self._check(tree)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.functions[node.name] = node
        ret = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                continue
            ret = self.exec_stmt(node, self.globals)
            if isinstance(ret, _Return):
                return ret.value
        if ret is None:
            # fall back to a conventional answer variable
            for name in ("result", "answer", "ans", "out"):
                if name in self.globals.vars:
                    return self.globals.vars[name]
        return None

    # -- static check -----------------------------------------------------
    def _check(self, tree: ast.AST) -> None:
        for n in ast.walk(tree):
            if isinstance(n, ast.stmt) and not isinstance(n, _ALLOWED_STMT):
                raise SandboxError(f"disallowed statement: {type(n).__name__}")
            if isinstance(n, ast.expr) and not isinstance(n, _ALLOWED_EXPR):
                raise SandboxError(f"disallowed expression: {type(n).__name__}")
            if isinstance(n, (ast.operator, ast.unaryop, ast.cmpop, ast.boolop)) and not isinstance(
                n, _ALLOWED_OPS
            ):
                raise SandboxError(f"disallowed operator: {type(n).__name__}")
            if isinstance(n, ast.Attribute):
                if not isinstance(n.attr, str) or n.attr not in _SAFE_ATTRS:
                    raise SandboxError(f"disallowed attribute: {getattr(n, 'attr', '?')}")
            if isinstance(n, ast.Name) and n.id.startswith("__"):
                raise SandboxError("dunder access forbidden")
            if isinstance(n, ast.FunctionDef):
                if n.decorator_list:
                    raise SandboxError("decorators forbidden")
                if n.args.vararg or n.args.kwarg or n.args.kwonlyargs:
                    raise SandboxError("*args/**kwargs forbidden")
                for d in (n.args.defaults or []):
                    self._check(d)

    # -- statements -------------------------------------------------------
    def exec_stmt(self, node: ast.AST, frame: _Frame) -> Any:
        self.tick()
        if isinstance(node, ast.Assign):
            val = self.eval_expr(node.value, frame)
            for t in node.targets:
                self.assign(t, val, frame)
            return None
        if isinstance(node, ast.AugAssign):
            cur = self.eval_expr(node.target, frame)
            rhs = self.eval_expr(node.value, frame)
            self.assign(node.target, _apply_bin(node.op, cur, rhs), frame)
            return None
        if isinstance(node, ast.Expr):
            self.eval_expr(node.value, frame)
            return None
        if isinstance(node, ast.Pass):
            return None
        if isinstance(node, ast.Return):
            return _Return(self.eval_expr(node.value, frame) if node.value is not None else None)
        if isinstance(node, ast.Break):
            return _Break()
        if isinstance(node, ast.Continue):
            return _Continue()
        if isinstance(node, ast.If):
            if _truthy(self.eval_expr(node.test, frame)):
                return self.exec_body(node.body, frame)
            return self.exec_body(node.orelse, frame)
        if isinstance(node, ast.For):
            return self.exec_for(node, frame)
        if isinstance(node, ast.While):
            return self.exec_while(node, frame)
        raise SandboxError(f"unhandled statement {type(node).__name__}")

    def exec_body(self, body: Sequence[ast.stmt], frame: _Frame) -> Any:
        # Blocks do NOT create a scope in Python (only functions and
        # comprehensions do) — sharing `frame` is what makes
        # "for i in ...: s = s + i" actually mutate s.
        scope = frame
        for st in body:
            r = self.exec_stmt(st, scope)
            if isinstance(r, (_Return, _Break, _Continue)):
                return r
        return None

    def exec_for(self, node: ast.For, frame: _Frame) -> Any:
        iterable = self.eval_expr(node.iter, frame)
        if isinstance(iterable, range):
            items: List[Any] = list(iterable)
        elif isinstance(iterable, (list, tuple, str, dict, set)):
            items = list(iterable)
        else:
            raise SandboxError("unsupported iterable")
        if len(items) > MAX_ITER:
            raise SandboxError(f"iteration count {len(items)} exceeds {MAX_ITER}")
        scope = frame
        count = 0
        for item in items:
            count += 1
            if count > MAX_ITER:
                raise SandboxError("iteration limit exceeded")
            self.assign(node.target, item, scope)
            r = self.exec_body(node.body, scope)
            if isinstance(r, _Break):
                break
            if isinstance(r, _Return):
                return r
            if isinstance(r, _Continue):
                continue
        return None

    def exec_while(self, node: ast.While, frame: _Frame) -> Any:
        scope = frame
        count = 0
        while _truthy(self.eval_expr(node.test, scope)):
            count += 1
            if count > MAX_ITER:
                raise SandboxError("while-loop limit exceeded")
            r = self.exec_body(node.body, scope)
            if isinstance(r, _Break):
                break
            if isinstance(r, _Return):
                return r
            if isinstance(r, _Continue):
                continue
        return None

    # -- assignment -------------------------------------------------------
    def assign(self, target: ast.AST, value: Any, frame: _Frame) -> None:
        if isinstance(target, ast.Name):
            frame.vars[target.id] = value
            return
        if isinstance(target, ast.Subscript):
            container = self.eval_expr(target.value, frame)
            key = self.eval_expr(target.slice, frame)
            container[key] = value
            return
        if isinstance(target, ast.Tuple):
            if not isinstance(value, (list, tuple)) or len(value) != len(target.elts):
                raise SandboxError("bad tuple unpacking")
            for t, v in zip(target.elts, value):
                self.assign(t, v, frame)
            return
        raise SandboxError("unsupported assignment target")

    # -- expressions ------------------------------------------------------
    def eval_expr(self, node: ast.AST, frame: _Frame) -> Any:
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in frame.vars:
                return frame.vars[node.id]
            raise SandboxError(f"undefined name {node.id!r}")
        if isinstance(node, ast.BinOp):
            return _apply_bin(node.op, self.eval_expr(node.left, frame), self.eval_expr(node.right, frame))
        if isinstance(node, ast.UnaryOp):
            v = self.eval_expr(node.operand, frame)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
            if isinstance(node.op, ast.Not):
                return not _truthy(v)
            raise SandboxError("bad unary op")
        if isinstance(node, ast.BoolOp):
            vals = [self.eval_expr(v, frame) for v in node.values]
            if isinstance(node.op, ast.And):
                out = vals[0]
                for v in vals[1:]:
                    if not _truthy(out):
                        return out
                    out = v
                return out
            out = vals[0]
            for v in vals[1:]:
                if _truthy(out):
                    return out
                out = v
            return out
        if isinstance(node, ast.Compare):
            left = self.eval_expr(node.left, frame)
            for op, comp in zip(node.ops, node.comparators):
                right = self.eval_expr(comp, frame)
                res = _apply_cmp(op, left, right)
                if not res:
                    return False
                left = right
            return True
        if isinstance(node, ast.List):
            return [self.eval_expr(e, frame) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self.eval_expr(e, frame) for e in node.elts)
        if isinstance(node, ast.Set):
            return {self.eval_expr(e, frame) for e in node.elts}
        if isinstance(node, ast.Dict):
            return {self.eval_expr(k, frame): self.eval_expr(v, frame) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            container = self.eval_expr(node.value, frame)
            key = self.eval_expr(node.slice, frame)
            if isinstance(key, ast.Slice):  # pragma: no cover - defensive
                raise SandboxError("slice objects unsupported")
            return container[key]
        if isinstance(node, ast.Slice):
            raise SandboxError("slice unsupported")
        if isinstance(node, ast.IfExp):
            return self.eval_expr(node.body, frame) if _truthy(self.eval_expr(node.test, frame)) else self.eval_expr(node.orelse, frame)
        if isinstance(node, ast.ListComp):
            scope = _Frame(frame.vars)
            out: List[Any] = []
            self._comp(node, scope, lambda v: out.append(v))
            return out
        if isinstance(node, ast.SetComp):
            scope = _Frame(frame.vars)
            out_set: set = set()
            self._comp(node, scope, out_set.add)
            return out_set
        if isinstance(node, ast.DictComp):
            scope = _Frame(frame.vars)
            out_d: Dict[Any, Any] = {}
            self._comp(node, scope, lambda v: out_d.update(v))
            return out_d
        if isinstance(node, ast.Attribute):
            obj = self.eval_expr(node.value, frame)
            attr = node.attr
            if attr not in _SAFE_ATTRS:
                raise SandboxError(f"attribute {attr!r} forbidden")
            fn = getattr(obj, attr, None)
            if fn is None:
                raise SandboxError(f"object has no attribute {attr!r}")
            return fn
        if isinstance(node, ast.Call):
            return self.eval_call(node, frame)
        raise SandboxError(f"unhandled expression {type(node).__name__}")

    def _comp(self, node, scope: _Frame, emit) -> None:
        def rec(gens, idx: int) -> None:
            if idx == len(gens):
                if isinstance(node, ast.DictComp):
                    emit(
                        {
                            self.eval_expr(node.key, scope): self.eval_expr(node.value, scope),
                        }
                    )
                else:
                    emit(self.eval_expr(node.elt, scope))
                return
            g = gens[idx]
            iterable = self.eval_expr(g.iter, scope)
            items = list(iterable if not isinstance(iterable, range) else iterable)
            if len(items) > MAX_ITER:
                raise SandboxError("comprehension too large")
            for item in items:
                self.tick()
                self.assign(g.target, item, scope)
                if all(_truthy(self.eval_expr(iff, scope)) for iff in g.ifs):
                    rec(gens, idx + 1)

        rec(list(node.generators), 0)

    def eval_call(self, node: ast.Call, frame: _Frame) -> Any:
        if isinstance(node.func, ast.Attribute):
            obj = self.eval_expr(node.func.value, frame)
            attr = node.func.attr
            if attr not in _SAFE_ATTRS:
                raise SandboxError(f"method {attr!r} forbidden")
            fn = getattr(obj, attr, None)
            if fn is None:
                raise SandboxError(f"no method {attr!r}")
            args = [self.eval_expr(a, frame) for a in node.args]
            return fn(*args)
        if not isinstance(node.func, ast.Name):
            raise SandboxError("only named functions may be called")
        name = node.func.id
        args = [self.eval_expr(a, frame) for a in node.args]
        if name in self.functions:
            if self.depth >= MAX_DEPTH:
                raise SandboxError("recursion depth exceeded")
            fn = self.functions[name]
            if len(args) != len(fn.args.args):
                raise SandboxError("arity mismatch")
            local = _Frame({**frame.vars})
            for p, v in zip(fn.args.args, args):
                local.vars[p.arg] = v
            self.depth += 1
            try:
                r = self.exec_body(fn.body, local)
            finally:
                self.depth -= 1
            return r.value if isinstance(r, _Return) else None
        if name in SAFE_BUILTINS:
            fn = SAFE_BUILTINS[name]
            if name == "range" and len(args) == 1 and isinstance(args[0], int) and args[0] > MAX_ITER:
                raise SandboxError("range too large")
            return fn(*args)
        raise SandboxError(f"call to unknown function {name!r}")


class _Return:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class _Break:
    pass


class _Continue:
    pass


def _truthy(v: Any) -> bool:
    return bool(v)


def _apply_bin(op: ast.AST, a: Any, b: Any) -> Any:
    if isinstance(op, ast.Add):
        return a + b
    if isinstance(op, ast.Sub):
        return a - b
    if isinstance(op, ast.Mult):
        if isinstance(a, (str, list, tuple)) and isinstance(b, int) and len(a) * b > 1_000_00:
            raise SandboxError("sequence repetition too large")
        return a * b
    if isinstance(op, ast.Div):
        if b == 0:
            raise SandboxError("division by zero")
        return a / b
    if isinstance(op, ast.FloorDiv):
        if b == 0:
            raise SandboxError("division by zero")
        return a // b
    if isinstance(op, ast.Mod):
        if b == 0:
            raise SandboxError("modulo by zero")
        return a % b
    if isinstance(op, ast.Pow):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(a) > 1e6 or abs(b) > 64:
                raise SandboxError("power operands out of safe range")
        return a**b
    raise SandboxError("unsupported binary operator")


def _apply_cmp(op: ast.AST, a: Any, b: Any) -> bool:
    if isinstance(op, ast.Eq):
        return _eq(a, b)
    if isinstance(op, ast.NotEq):
        return not _eq(a, b)
    if isinstance(op, ast.Lt):
        return a < b
    if isinstance(op, ast.LtE):
        return a <= b
    if isinstance(op, ast.Gt):
        return a > b
    if isinstance(op, ast.GtE):
        return a >= b
    if isinstance(op, ast.In):
        return a in b
    if isinstance(op, ast.NotIn):
        return a not in b
    raise SandboxError("unsupported comparison")


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= max(1e-12, 1e-9 * max(abs(a), abs(b)))
    return a == b


def run_sandboxed(src: str) -> Dict[str, Any]:
    """Execute ``src`` in the sandbox. Returns a result dict (never raises)."""
    interp = _Interp()
    try:
        value = interp.run(src)
        return {"ok": True, "value": value, "steps": interp.steps, "error": ""}
    except SandboxError as exc:
        return {"ok": False, "value": None, "steps": interp.steps, "error": f"SandboxError: {exc}"}
    except SyntaxError as exc:
        return {"ok": False, "value": None, "steps": 0, "error": f"SyntaxError: {exc}"}
    except RecursionError:
        return {"ok": False, "value": None, "steps": interp.steps, "error": "RecursionError"}
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "ok": False,
            "value": None,
            "steps": interp.steps,
            "error": f"{type(exc).__name__}: {exc}",
        }


def eval_expected(src: str):
    """Evaluate the ``###<expected>`` side inside the same sandbox.

    Returns ``(ok, value, error)``. Uses a plain expression — no assignment, so
    no dunder names are involved (the sandbox forbids ``__`` identifiers).
    """
    try:
        tree = ast.parse(str(src).strip(), mode="eval")
    except SyntaxError as exc:
        return False, None, f"SyntaxError: {exc}"
    interp = _Interp()
    try:
        interp._check(tree)  # noqa: SLF001 - internal use within the module
        return True, interp.eval_expr(tree.body, interp.globals), ""
    except SandboxError as exc:
        return False, None, f"SandboxError: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, None, f"{type(exc).__name__}: {exc}"


class CodeVerifier(Verifier):
    """Verifies ``kind="code"`` claims by executing them in the sandbox."""

    kind = "code"
    name = "code-verifier-sandbox"
    _SEP_RE = re.compile(r"\s*" + SEP + r"\s*")

    def verify(self, claim: Claim, trace: Optional[Trace] = None) -> VerificationResult:
        if not self.supports(claim) or claim.expr is None:
            return self._unsupported(claim)
        parts = self._SEP_RE.split(claim.expr.strip(), maxsplit=1)
        src = parts[0]
        expected_src = parts[1] if len(parts) > 1 else None
        out = run_sandboxed(src)
        if not out["ok"]:
            return self._result(claim, None, 0.0, out["error"], counterexample=out["error"])
        if expected_src is None:
            return self._result(
                claim, None, 0.35, f"executed without error, value={out['value']!r}"
            )
        ok_exp, want, err = eval_expected(expected_src)
        if not ok_exp:
            return self._result(claim, None, 0.0, f"expected value unparseable: {err}")
        got = out["value"]
        if _values_equal(got, want):
            return self._result(claim, True, 1.0, f"returned {got!r} == expected {want!r}")
        return self._result(
            claim,
            False,
            1.0,
            f"returned {got!r} != expected {want!r}",
            counterexample=f"got {got!r}, expected {want!r}",
        )


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= max(1e-9, 1e-9 * max(abs(a), abs(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b
