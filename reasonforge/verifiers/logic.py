"""
Logic verifier (命题逻辑验证器) — classical DPLL over a parsed propositional AST.

Supports two verification modes for ``kind="logic"`` claims:

1. **Entailment**: ``"A, A -> B |= B"`` (premises left of ``|=`` / ``⊨``)
2. **Tautology**: a bare formula asserted valid (checked via refutation of its negation)

Pipeline: tokenise -> recursive-descent parse -> AST -> CNF (Tseitin)
-> DPLL with unit propagation. All classical, all deterministic.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from ..core.trace import Claim, Trace
from .base import Verifier, VerificationResult

__all__ = [
    "LogicVerifier",
    "LogicError",
    "parse_formula",
    "to_cnf",
    "dpll",
    "satisfiable",
    "valid",
    "entails",
    "find_model",
]

# --------------------------------------------------------------------------
# Tokens & parser
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<LP>\()|(?P<RP>\))|"
    r"(?P<IFF><->|<=>|↔|≡)|"
    r"(?P<IMP>->|=>|→|⊃)|"
    r"(?P<AND>&&|&|∧|and|AND)|"
    r"(?P<OR>\|\||\||∨|or|OR)|"
    r"(?P<NOT>~|!|¬|not|NOT)|"
    r"(?P<TOP>⊤|True|TRUE|true)|"
    r"(?P<BOT>⊥|False|FALSE|false)|"
    r"(?P<ID>[A-Za-z_][A-Za-z0-9_]*)"
    r")"
)


class LogicError(Exception):
    """Raised for malformed propositional formulas."""


@dataclass(frozen=True)
class Node:
    op: str  # 'var' | 'not' | 'and' | 'or' | 'imp' | 'iff' | 'top' | 'bot'
    name: str = ""
    kids: Tuple["Node", ...] = ()


def _tokenize(src: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    i = 0
    n = len(src)
    while i < n:
        if src[i].isspace():
            i += 1
            continue
        if src[i] == ",":
            # "p, q" is a premise list: treat the comma as conjunction
            out.append(("AND", ","))
            i += 1
            continue
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise LogicError(f"cannot tokenize at position {i}: {src[i:][:12]!r}")
        kind = m.lastgroup or "?"
        out.append((kind, m.group()))
        i = m.end()
    return out


class _Parser:
    """Grammar (lowest precedence last):

    iff := imp (IFF imp)*
    imp := or  (IMP or)*          (right associative)
    or  := and (OR and)*
    and := unary (AND unary)*
    unary := NOT unary | atom
    atom := ID | TOP | BOT | '(' iff ')'
    """

    def __init__(self, tokens: Sequence[Tuple[str, str]]) -> None:
        self.t = list(tokens)
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.t[self.i][0] if self.i < len(self.t) else None

    def eat(self, kind: str) -> str:
        if self.peek() != kind:
            raise LogicError(f"expected {kind}, got {self.peek()}")
        v = self.t[self.i][1]
        self.i += 1
        return v

    def parse(self) -> Node:
        node = self.iff()
        if self.i != len(self.t):
            raise LogicError("trailing tokens")
        return node

    def iff(self) -> Node:
        left = self.imp()
        while self.peek() == "IFF":
            self.eat("IFF")
            left = Node("iff", kids=(left, self.imp()))
        return left

    def imp(self) -> Node:
        left = self.or_()
        if self.peek() == "IMP":
            self.eat("IMP")
            right = self.imp()  # right associative
            return Node("imp", kids=(left, right))
        return left

    def or_(self) -> Node:
        left = self.and_()
        while self.peek() == "OR":
            self.eat("OR")
            left = Node("or", kids=(left, self.and_()))
        return left

    def and_(self) -> Node:
        left = self.unary()
        while self.peek() == "AND":
            self.eat("AND")
            left = Node("and", kids=(left, self.unary()))
        return left

    def unary(self) -> Node:
        if self.peek() == "NOT":
            self.eat("NOT")
            return Node("not", kids=(self.unary(),))
        return self.atom()

    def atom(self) -> Node:
        k = self.peek()
        if k == "TOP":
            self.eat("TOP")
            return Node("top")
        if k == "BOT":
            self.eat("BOT")
            return Node("bot")
        if k == "ID":
            return Node("var", name=self.eat("ID"))
        if k == "LP":
            self.eat("LP")
            inner = self.iff()
            self.eat("RP")
            return inner
        raise LogicError(f"unexpected token {k}")


def parse_formula(src: str) -> Node:
    """Parse a propositional formula. Raises :class:`LogicError` on bad input."""
    return _Parser(_tokenize(src)).parse()


def variables(node: Node) -> List[str]:
    out: List[str] = []

    def walk(n: Node) -> None:
        if n.op == "var":
            if n.name not in out:
                out.append(n.name)
        for k in n.kids:
            walk(k)

    walk(node)
    return out


# --------------------------------------------------------------------------
# CNF via Tseitin
# --------------------------------------------------------------------------
def to_cnf(node: Node) -> Tuple[List[Tuple[int, ...]], Dict[str, int]]:
    """Tseitin encoding -> ``(clauses, var_index)``.

    Literals are integers: ``v`` positive, ``-v`` negative. Variable 0 is
    reserved and never used. Fresh auxiliary variables are added per gate.

    Invariant: ``satisfiable(to_cnf(n)) == satisfiable(n)`` (equisatisfiable).
    """
    clauses: List[Tuple[int, ...]] = []
    index: Dict[str, int] = {}
    counter = [0]

    def fresh() -> int:
        counter[0] += 1
        return counter[0]

    def vid(name: str) -> int:
        if name not in index:
            index[name] = fresh()
        return index[name]

    def enc(n: Node) -> int:
        if n.op == "var":
            return vid(n.name)
        if n.op == "top":
            t = fresh()
            clauses.append((t,))
            return t
        if n.op == "bot":
            b = fresh()
            clauses.append((-b,))
            return b
        if n.op == "not":
            a = enc(n.kids[0])
            o = fresh()
            clauses.append((-o, -a))
            clauses.append((o, a))
            return o
        if n.op == "and":
            kids = [enc(k) for k in n.kids]
            o = fresh()
            for a in kids:
                clauses.append((-o, a))
            clauses.append((o,) + tuple(-a for a in kids))
            return o
        if n.op == "or":
            kids = [enc(k) for k in n.kids]
            o = fresh()
            for a in kids:
                clauses.append((o, -a))
            clauses.append((-o,) + tuple(a for a in kids))
            return o
        if n.op == "imp":
            a = enc(n.kids[0])
            b = enc(n.kids[1])
            o = fresh()
            clauses.append((-o, -a, b))
            clauses.append((o, a))
            clauses.append((o, -b))
            return o
        if n.op == "iff":
            a = enc(n.kids[0])
            b = enc(n.kids[1])
            o = fresh()
            clauses.append((-o, -a, b))
            clauses.append((-o, a, -b))
            clauses.append((o, -a, -b))
            clauses.append((o, a, b))
            return o
        raise LogicError(f"unknown op {n.op}")

    root = enc(node)
    clauses.append((root,))
    return clauses, index


# --------------------------------------------------------------------------
# DPLL
# --------------------------------------------------------------------------
def dpll(clauses: Sequence[Sequence[int]]) -> Optional[Dict[int, bool]]:
    """Classical DPLL with unit propagation. Returns a model or ``None`` (UNSAT).

    Invariant: a returned model satisfies every clause; ``None`` implies UNSAT.
    """
    cls: List[FrozenSet[int]] = [frozenset(c) for c in clauses]
    assign: Dict[int, bool] = {}

    def simplify(cs: List[FrozenSet[int]], lit: int) -> Optional[List[FrozenSet[int]]]:
        out: List[FrozenSet[int]] = []
        neg = -lit
        for c in cs:
            if lit in c:
                continue
            if neg in c:
                c = frozenset(x for x in c if x != neg)
                if not c:
                    return None
            out.append(c)
        return out

    def unit_propagate(cs: List[FrozenSet[int]], asg: Dict[int, bool]) -> Optional[Tuple[List[FrozenSet[int]], Dict[int, bool]]]:
        changed = True
        while changed:
            changed = False
            if not cs:
                return cs, asg
            if any(not c for c in cs):
                return None
            for c in cs:
                if len(c) == 1:
                    lit = next(iter(c))
                    asg[abs(lit)] = lit > 0
                    nxt = simplify(cs, lit)
                    if nxt is None:
                        return None
                    cs = nxt
                    changed = True
                    break
        return cs, asg

    def solve(cs: List[FrozenSet[int]], asg: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        r = unit_propagate(cs, asg)
        if r is None:
            return None
        cs, asg = r
        if not cs:
            return asg
        if any(not c for c in cs):
            return None
        # pure literal elimination (sound & cheap)
        counts: Dict[int, int] = {}
        for c in cs:
            for lit in c:
                counts[lit] = counts.get(lit, 0) + 1
        for lit in sorted(counts, key=lambda x: (abs(x), x)):
            if -lit not in counts:
                nxt = simplify(cs, lit)
                if nxt is not None:
                    a2 = dict(asg)
                    a2[abs(lit)] = lit > 0
                    res = solve(nxt, a2)
                    if res is not None:
                        return res
                break
        lit = next(iter(sorted(cs[0], key=lambda x: (abs(x), x))))
        for candidate, val in ((lit, True), (-lit, False)):
            nxt = simplify(cs, candidate)
            if nxt is None:
                continue
            a2 = dict(asg)
            a2[abs(candidate)] = val
            res = solve(nxt, a2)
            if res is not None:
                return res
        return None

    return solve(cls, assign)


def find_model(node_or_src) -> Optional[Dict[str, bool]]:
    """Satisfying assignment over *named* variables (aux vars hidden)."""
    node = parse_formula(node_or_src) if isinstance(node_or_src, str) else node_or_src
    clauses, index = to_cnf(node)
    model = dpll(clauses)
    if model is None:
        return None
    rev = {v: k for k, v in index.items()}
    out = {rev[v]: b for v, b in model.items() if v in rev}
    for name in variables(node):
        out.setdefault(name, False)
    return out


def satisfiable(node_or_src) -> bool:
    node = parse_formula(node_or_src) if isinstance(node_or_src, str) else node_or_src
    return dpll(to_cnf(node)[0]) is not None


def valid(node_or_src) -> bool:
    """Tautology check: ``valid(F) <=> UNSAT(¬F)``."""
    node = parse_formula(node_or_src) if isinstance(node_or_src, str) else node_or_src
    return not satisfiable(Node("not", kids=(node,)))


def entails(premises: Sequence, conclusion) -> bool:
    """``premises ⊨ conclusion``  <=>  ``valid(∧premises -> conclusion)``."""
    ps = [parse_formula(p) if isinstance(p, str) else p for p in premises]
    c = parse_formula(conclusion) if isinstance(conclusion, str) else conclusion
    if not ps:
        return valid(c)
    conj = ps[0]
    for p in ps[1:]:
        conj = Node("and", kids=(conj, p))
    return valid(Node("imp", kids=(conj, c)))


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------
class LogicVerifier(Verifier):
    """Verifies ``kind="logic"`` claims.

    Accepted ``expr`` forms::

        "A, A -> B |= B"      premises entail conclusion
        "A, A -> B ⊨ B"
        "A or not A"          bare formula, checked for validity (tautology)
    """

    kind = "logic"
    name = "logic-verifier-dpll"

    _ENTAIL_RE = re.compile(r"\s*(?:\|=|⊨|=>\||therefore|THEREFORE)\s*")
    _MODE_RE = re.compile(
        r"^\s*(?P<mode>SAT|UNSAT|VALID|INVALID)\s*[:：]\s*(?P<body>.*)$", re.IGNORECASE
    )

    def verify(self, claim: Claim, trace: Optional[Trace] = None) -> VerificationResult:
        if not self.supports(claim) or claim.expr is None:
            return self._unsupported(claim)
        expr = claim.expr.strip()

        # --- explicit mode prefixes: SAT / UNSAT / VALID / INVALID --------
        m = self._MODE_RE.match(expr)
        if m:
            return self._verify_mode(claim, m.group("mode").upper(), m.group("body"))

        try:
            parts = self._ENTAIL_RE.split(expr)
            if len(parts) >= 2:
                premises = [p for p in re.split(r"[;,\n]+", parts[0]) if p.strip()]
                conclusion = parts[-1]
                ok = entails(premises, conclusion)
                if ok:
                    return self._result(claim, True, 1.0, "entailment holds (refutation UNSAT)")
                model = self._countermodel(premises, conclusion)
                ce = ""
                if model:
                    ce = "countermodel: " + ", ".join(
                        f"{k}={'T' if v else 'F'}" for k, v in sorted(model.items())
                    )
                return self._result(
                    claim, False, 1.0, "premises do not entail conclusion", counterexample=ce or None
                )
            ok = valid(expr)
            if ok:
                return self._result(claim, True, 1.0, "formula is a tautology")
            m = find_model(Node("not", kids=(parse_formula(expr),)))
            ce = None
            if m:
                ce = "falsifying assignment: " + ", ".join(
                    f"{k}={'T' if v else 'F'}" for k, v in sorted(m.items())
                )
            return self._result(claim, False, 1.0, "formula is not valid", counterexample=ce)
        except LogicError as exc:
            return self._result(claim, None, 0.0, f"parse error: {exc}")
        except RecursionError:
            return self._result(claim, None, 0.0, "formula too deep")

    def _verify_mode(self, claim: Claim, mode: str, body: str) -> VerificationResult:
        """Handle ``SAT:`` / ``UNSAT:`` / ``VALID:`` / ``INVALID:`` prefixed claims."""
        try:
            node = parse_formula(body)
        except LogicError as exc:
            return self._result(claim, None, 0.0, f"parse error: {exc}")
        try:
            sat = satisfiable(node)
            val = valid(node)
        except RecursionError:
            return self._result(claim, None, 0.0, "formula too deep")

        def model_str(target: Node) -> Optional[str]:
            m = find_model(target)
            if not m:
                return None
            return ", ".join(f"{k}={'T' if v else 'F'}" for k, v in sorted(m.items()))

        if mode == "SAT":
            if sat:
                return self._result(claim, True, 1.0, f"satisfiable (model: {model_str(node)})")
            return self._result(claim, False, 1.0, "unsatisfiable",
                                counterexample="no model exists")
        if mode == "UNSAT":
            if not sat:
                return self._result(claim, True, 1.0, "unsatisfiable as asserted")
            ce = model_str(node)
            return self._result(claim, False, 1.0, "formula is satisfiable",
                                counterexample=f"model: {ce}")
        if mode == "VALID":
            if val:
                return self._result(claim, True, 1.0, "valid (tautology)")
            ce = model_str(Node("not", kids=(node,)))
            return self._result(claim, False, 1.0, "not valid",
                                counterexample=f"falsifying assignment: {ce}")
        # INVALID: asserted *not* valid
        if not val:
            ce = model_str(Node("not", kids=(node,)))
            return self._result(claim, True, 1.0, "not valid as asserted",
                                counterexample=None)
        return self._result(claim, False, 1.0, "formula is actually valid",
                            counterexample="no falsifying assignment exists")

    @staticmethod
    def _countermodel(premises: Sequence[str], conclusion: str) -> Optional[Dict[str, bool]]:
        try:
            ps = [parse_formula(p) for p in premises]
            conj = ps[0]
            for p in ps[1:]:
                conj = Node("and", kids=(conj, p))
            neg = Node("and", kids=(conj, Node("not", kids=(parse_formula(conclusion),))))
            return find_model(neg)
        except LogicError:
            return None


def _free_vars_set(node: Node) -> Set[str]:
    return set(variables(node))
