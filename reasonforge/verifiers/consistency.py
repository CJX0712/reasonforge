"""
Consistency verifier (一致性验证器) — intra-trace and cross-trace agreement.

Two distinct jobs:

1. :class:`ConsistencyVerifier` — claim-level: does the trace's final answer
   agree with the value this claim asserts? Catches the classic failure mode
   where a chain computes ``60`` in the body but emits ``80`` in the header.
2. :func:`cross_trace_conflicts` / :func:`answer_distribution` — system-level:
   where do independent chains contradict each other, and how is the answer
   mass distributed? Feeds the VWSC aggregator.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.numeric import entropy, normalized_entropy
from ..core.trace import Claim, Trace, answer_similarity, normalize_answer
from .base import Verifier, VerificationReport, VerificationResult

__all__ = [
    "ConsistencyVerifier",
    "cross_trace_conflicts",
    "answer_distribution",
    "agreement_entropy",
]


class ConsistencyVerifier(Verifier):
    """Verifies ``kind="consistency"`` claims against the trace's own answer.

    ``expr`` is the expected answer (e.g. ``"60"``). If the trace's answer
    normalises to the same key the claim passes; otherwise it fails with the
    mismatch as its counterexample.
    """

    kind = "consistency"
    name = "consistency-verifier"

    def verify(self, claim: Claim, trace: Optional[Trace] = None) -> VerificationResult:
        if not self.supports(claim) or claim.expr is None:
            return self._unsupported(claim)
        if trace is None:
            return self._result(claim, None, 0.0, "no trace context for consistency check")
        want_key = normalize_answer(claim.expr)
        got_key = normalize_answer(trace.answer)
        if want_key == got_key:
            return self._result(claim, True, 1.0, f"answer {got_key!r} matches claim")
        sim = answer_similarity(claim.expr, trace.answer)
        if sim >= 0.95:
            return self._result(claim, True, 0.8, f"near-match (sim={sim:.3f}): {got_key!r} ~ {want_key!r}")
        return self._result(
            claim,
            False,
            1.0,
            f"answer {got_key!r} contradicts claim {want_key!r}",
            counterexample=f"trace says {trace.answer!r}, claim asserts {claim.expr!r}",
        )


def answer_distribution(traces: Sequence[Trace]) -> Dict[str, int]:
    """Canonical answer -> count, for the sampled chains."""
    c: Counter = Counter(t.key() for t in traces)
    return dict(c)


def agreement_entropy(traces: Sequence[Trace]) -> float:
    """Normalised entropy of the answer distribution in [0,1].

    0 -> all chains agree (no need for more compute);
    1 -> maximum disagreement (spend more compute, or trust verifiers).
    """
    dist = answer_distribution(traces)
    if len(dist) <= 1:
        return 0.0
    counts = list(dist.values())
    return normalized_entropy([float(x) for x in counts], base=2.0)


def cross_trace_conflicts(
    traces: Sequence[Trace], reports: Optional[Dict[str, VerificationReport]] = None
) -> List[Dict[str, object]]:
    """Detect identical claims verified differently in different traces.

    Also flags *semantic* conflicts: two traces with the same claim id but
    different ``expr`` payloads, or the same expr with opposite verdicts.
    """
    reports = reports or {}
    by_id: Dict[str, List[Tuple[str, str, Optional[bool]]]] = {}
    for t in traces:
        rep = reports.get(t.id)
        for c in t.claims():
            if not c.expr:
                continue
            ok = rep.results[c.id].ok if rep and c.id in rep.results else None
            by_id.setdefault(c.id, []).append((t.id, c.expr, ok))

    conflicts: List[Dict[str, object]] = []
    for cid, entries in by_id.items():
        exprs = {e[1] for e in entries}
        verdicts = {e[2] for e in entries if e[2] is not None}
        if len(exprs) > 1:
            conflicts.append(
                {
                    "claim_id": cid,
                    "type": "payload-conflict",
                    "variants": sorted(exprs),
                    "traces": [e[0] for e in entries],
                }
            )
        elif len(verdicts) > 1:
            conflicts.append(
                {
                    "claim_id": cid,
                    "type": "verdict-conflict",
                    "expr": next(iter(exprs)),
                    "verdicts": sorted({str(v) for v in verdicts}),
                    "traces": [e[0] for e in entries],
                }
            )
    conflicts.sort(key=lambda d: str(d.get("claim_id")))
    return conflicts
