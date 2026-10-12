"""
L1 — Reasoning trace structures (推理链表示层)。

A reasoning trace is a **DAG of claims**: every step asserts one or more
verifiable claims, and each claim may depend on earlier claims. This is the
data structure that makes verification possible at all — an unstructured
"answer string" cannot be checked, a claim graph can.

Design contract
---------------
* ``Claim.expr`` is the machine-checkable payload (e.g. ``"12*5 = 60"``).
* ``Claim.deps`` references prior claim ids -> forms a DAG.
* :meth:`Trace.validate` asserts **referential integrity + acyclicity**.
* :func:`normalize_answer` provides the canonical key used for voting, so that
  ``"60"``, ``"60.0"``, ``" 60 "`` and ``"$60"`` collapse to one bucket.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Claim",
    "Step",
    "Trace",
    "TraceError",
    "normalize_answer",
    "answer_similarity",
]

CLAIM_KINDS = (
    "arithmetic",  # 算术：可用表达式求值器判定
    "algebra",     # 代数/符号：可用 sympy 或内置多项式检查
    "logic",       # 命题逻辑：DPLL/SAT
    "constraint",  # 约束满足：CSP 传播
    "code",        # 可执行代码：沙箱执行
    "fact",        # 事实/检索：需外部证据
    "consistency",  # 一致性：与给定答案/其它链交叉核对
    "common",      # 常识：不可机器判定，标记为 unverifiable
)

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_STRIP_RE = re.compile(r"[,\s$€¥£%度元美元]+")


class TraceError(ValueError):
    """Raised when a trace violates its structural contract."""


@dataclass(frozen=True)
class Claim:
    """A single machine-checkable assertion inside a reasoning trace."""

    id: str
    text: str
    kind: str = "common"
    expr: Optional[str] = None
    deps: Tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in CLAIM_KINDS:
            raise TraceError(f"unknown claim kind: {self.kind!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise TraceError("confidence must be in [0, 1]")
        seen: Set[str] = set()
        for d in self.deps:
            if d in seen:
                raise TraceError(f"duplicate dependency {d!r} in claim {self.id!r}")
            seen.add(d)
            if d == self.id:
                raise TraceError(f"claim {self.id!r} depends on itself")

    def verifiable(self) -> bool:
        """Whether a machine verifier exists for this claim kind."""
        return self.kind in ("arithmetic", "algebra", "logic", "constraint", "code", "consistency")


@dataclass
class Step:
    """One reasoning step: natural-language text plus its asserted claims."""

    id: str
    text: str
    claims: List[Claim] = field(default_factory=list)
    score: float = 1.0  # process reward signal in [0,1]

    def claim_ids(self) -> List[str]:
        return [c.id for c in self.claims]


@dataclass
class Trace:
    """A complete chain of thought, answerable and verifiable."""

    id: str
    steps: List[Step] = field(default_factory=list)
    answer: str = ""
    answer_claim: Optional[str] = None
    logprob: float = 0.0
    meta: Dict[str, object] = field(default_factory=dict)

    # -- structure ---------------------------------------------------------
    def claims(self) -> List[Claim]:
        out: List[Claim] = []
        for s in self.steps:
            out.extend(s.claims)
        return out

    def claim_map(self) -> Dict[str, Claim]:
        return {c.id: c for c in self.claims()}

    def validate(self) -> None:
        """Assert referential integrity and acyclicity.

        Raises :class:`TraceError` on violation. This is the L1 hard invariant.
        """
        cmap = self.claim_map()
        ids = [c.id for c in self.claims()]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise TraceError(f"duplicate claim ids: {sorted(dupes)}")
        for c in self.claims():
            for d in c.deps:
                if d not in cmap:
                    raise TraceError(f"claim {c.id!r} depends on unknown claim {d!r}")
        # Kahn topological sort -> detects cycles
        indeg: Dict[str, int] = {c.id: len(c.deps) for c in self.claims()}
        adj: Dict[str, List[str]] = {c.id: [] for c in self.claims()}
        for c in self.claims():
            for d in c.deps:
                adj[d].append(c.id)
        queue = [i for i, v in indeg.items() if v == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if seen != len(indeg):
            raise TraceError("claim dependency graph contains a cycle")
        if self.answer_claim is not None and self.answer_claim not in cmap:
            raise TraceError(f"answer_claim {self.answer_claim!r} not present in trace")

    def is_valid(self) -> bool:
        try:
            self.validate()
            return True
        except TraceError:
            return False

    def topo_order(self) -> List[Claim]:
        """Claims in dependency-respecting order (stable, ties by declaration)."""
        self.validate()
        cmap = self.claim_map()
        order = [c.id for c in self.claims()]
        pos = {cid: i for i, cid in enumerate(order)}
        indeg = {c.id: len(c.deps) for c in self.claims()}
        adj: Dict[str, List[str]] = {c.id: [] for c in self.claims()}
        for c in self.claims():
            for d in c.deps:
                adj[d].append(c.id)
        ready = sorted((i for i, v in indeg.items() if v == 0), key=lambda x: pos[x])
        out: List[Claim] = []
        while ready:
            n = ready.pop(0)
            out.append(cmap[n])
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
            ready.sort(key=lambda x: pos[x])
        return out

    def depth(self) -> int:
        """Longest dependency chain length (0 for independent claims)."""
        try:
            order = self.topo_order()
        except TraceError:
            return -1
        cmap = self.claim_map()
        best: Dict[str, int] = {}
        for c in order:
            best[c.id] = 0 if not c.deps else 1 + max(best[d] for d in c.deps)
        return (max(best.values()) + 1) if best else 0

    def n_steps(self) -> int:
        return len(self.steps)

    def key(self) -> str:
        """Canonical voting key of the final answer."""
        return normalize_answer(self.answer)

    def __len__(self) -> int:
        return len(self.steps)


# --------------------------------------------------------------------------
# Answer normalisation
# --------------------------------------------------------------------------
def normalize_answer(text: str) -> str:
    """Canonicalise an answer string so equivalent answers vote together.

    Handles: whitespace/currency/punctuation stripping, numeric canonicalisation
    (``60.0`` -> ``60``), case folding, and trailing-period removal.

    Invariant: ``normalize_answer`` is idempotent — ``f(f(x)) == f(x)``.
    """
    if text is None:
        return ""
    s = str(text).strip()
    s = _STRIP_RE.sub("", s)
    s = s.rstrip(".。!！?？;；")
    s = s.strip("`'\"“”‘’()[]{}")
    s = s.strip()
    if not s:
        return ""
    # Pure number -> canonical numeric form
    if _NUM_RE.fullmatch(s):
        try:
            v = float(s)
        except ValueError:
            return s.lower()
        if math.isfinite(v):
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return f"{v:.10g}"
    # Numeric ratio like 3/4
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            num, den = float(m.group(1)), float(m.group(2))
            if den != 0:
                v = num / den
                if abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                return f"{v:.10g}"
        except ValueError:
            pass
    return s.lower()


def answer_similarity(a: str, b: str) -> float:
    """Cheap similarity in [0,1] for near-miss answer clustering.

    Exact canonical match -> 1.0. Otherwise a token/char Jaccard blend, so that
    ``"60"`` vs ``"60 dollars"`` still lands close enough for soft clustering.
    """
    ka, kb = normalize_answer(a), normalize_answer(b)
    if ka == kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    ta, tb = set(re.findall(r"\w+", ka)), set(re.findall(r"\w+", kb))
    if not ta or not tb:
        ca, cb = set(ka), set(kb)
        return len(ca & cb) / len(ca | cb)
    jac = len(ta & tb) / len(ta | tb)
    # numeric closeness
    na, nb = _first_number(ka), _first_number(kb)
    num = 0.0
    if na is not None and nb is not None:
        denom = max(abs(na), abs(nb), 1e-12)
        num = max(0.0, 1.0 - abs(na - nb) / denom)
    return max(jac, 0.9 * num)


def _first_number(s: str) -> Optional[float]:
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None
