"""
L4 — Verification layer: base contracts (符号验证层 · 基础契约).

The whole point of ``reasonforge`` is that **a chain of thought is only as good
as its weakest checkable claim**. This module defines the verifier contract:

    Claim -> VerificationResult(ok, confidence, detail, counterexample)

``ok`` is tri-state:
* ``True``  — claim is provably correct under the verifier's semantics
* ``False`` — claim is provably wrong (a *counterexample* is attached when possible)
* ``None``  — unverifiable (no machine semantics); must never be silently True

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..core.trace import Claim, Trace

__all__ = [
    "VerificationResult",
    "Verifier",
    "VerifierRegistry",
    "VerificationReport",
]


@dataclass
class VerificationResult:
    """Outcome of checking one claim."""

    claim_id: str
    kind: str
    ok: Optional[bool]
    confidence: float = 1.0
    detail: str = ""
    counterexample: Optional[str] = None
    checker: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")

    @property
    def passed(self) -> bool:
        return self.ok is True

    @property
    def failed(self) -> bool:
        return self.ok is False

    @property
    def unknown(self) -> bool:
        return self.ok is None

    def as_dict(self) -> Dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "kind": self.kind,
            "ok": self.ok,
            "confidence": self.confidence,
            "detail": self.detail,
            "counterexample": self.counterexample,
            "checker": self.checker,
        }


class Verifier:
    """Abstract verifier. Subclasses implement :meth:`verify`."""

    kind: str = "generic"
    name: str = "verifier"

    def supports(self, claim: Claim) -> bool:
        return claim.kind == self.kind and claim.expr is not None

    def verify(self, claim: Claim, trace: Optional[Trace] = None) -> VerificationResult:
        raise NotImplementedError

    # convenience ---------------------------------------------------------
    def _result(
        self,
        claim: Claim,
        ok: Optional[bool],
        confidence: float = 1.0,
        detail: str = "",
        counterexample: Optional[str] = None,
    ) -> VerificationResult:
        return VerificationResult(
            claim_id=claim.id,
            kind=claim.kind,
            ok=ok,
            confidence=confidence,
            detail=detail,
            counterexample=counterexample,
            checker=self.name,
        )

    def _unsupported(self, claim: Claim) -> VerificationResult:
        return self._result(claim, None, 0.0, "unsupported claim kind/expr")


class VerifierRegistry:
    """Routes claims to verifiers by kind and aggregates per-trace reports."""

    def __init__(self, verifiers: Optional[Sequence[Verifier]] = None) -> None:
        self._by_kind: Dict[str, List[Verifier]] = {}
        for v in verifiers or ():
            self.register(v)

    def register(self, verifier: Verifier) -> None:
        self._by_kind.setdefault(verifier.kind, []).append(verifier)

    def kinds(self) -> List[str]:
        return sorted(self._by_kind)

    def verify_claim(self, claim: Claim, trace: Optional[Trace] = None) -> VerificationResult:
        """Run all verifiers registered for the claim kind; take the strictest.

        Strictness order: ``False`` (hard failure) beats ``None`` beats ``True``.
        A single hard refutation is enough — majority voting is *wrong* here.
        """
        cands = [v for v in self._by_kind.get(claim.kind, []) if v.supports(claim)]
        if not cands:
            return VerificationResult(
                claim_id=claim.id,
                kind=claim.kind,
                ok=None,
                confidence=0.0,
                detail=f"no verifier registered for kind={claim.kind!r}",
                checker="registry",
            )
        results = [v.verify(claim, trace) for v in cands]
        for r in results:
            if r.failed:
                return r
        known = [r for r in results if r.passed]
        if known:
            known.sort(key=lambda r: -r.confidence)
            return known[0]
        return results[0]

    def verify_trace(self, trace: Trace) -> "VerificationReport":
        trace.validate()
        results = {c.id: self.verify_claim(c, trace) for c in trace.claims()}
        return VerificationReport(trace_id=trace.id, results=results)


@dataclass
class VerificationReport:
    """All verification results for one trace, plus derived diagnostics."""

    trace_id: str
    results: Dict[str, VerificationResult] = field(default_factory=dict)

    def passed(self) -> List[VerificationResult]:
        return [r for r in self.results.values() if r.passed]

    def failed(self) -> List[VerificationResult]:
        return [r for r in self.results.values() if r.failed]

    def unknown(self) -> List[VerificationResult]:
        return [r for r in self.results.values() if r.unknown]

    def n_passed(self) -> int:
        return len(self.passed())

    def n_failed(self) -> int:
        return len(self.failed())

    def n_unknown(self) -> int:
        return len(self.unknown())

    def counterexamples(self) -> List[str]:
        return [r.counterexample for r in self.failed() if r.counterexample]

    def as_dict(self) -> Dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "n_passed": self.n_passed(),
            "n_failed": self.n_failed(),
            "n_unknown": self.n_unknown(),
            "results": {k: v.as_dict() for k, v in self.results.items()},
        }
