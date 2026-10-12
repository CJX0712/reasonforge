"""
L1 invariants — reasoning trace structures.

The structural contract is what makes verification possible: a claim graph must
be a DAG with referential integrity, or the verifier cannot reason about it.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import pytest

from reasonforge.core.trace import (
    Claim,
    Step,
    Trace,
    TraceError,
    answer_similarity,
    normalize_answer,
)


def _claim(cid, deps=(), kind="arithmetic", expr="1 = 1"):
    return Claim(id=cid, text=f"claim {cid}", kind=kind, expr=expr, deps=deps)


# ------------------------------------------------------------------- claims
def test_claim_rejects_unknown_kind():
    with pytest.raises(TraceError):
        Claim(id="a", text="t", kind="not-a-kind")


def test_claim_rejects_self_dependency():
    with pytest.raises(TraceError):
        Claim(id="a", text="t", deps=("a",))


def test_claim_rejects_duplicate_dependency():
    with pytest.raises(TraceError):
        Claim(id="a", text="t", deps=("b", "b"))


def test_claim_rejects_bad_confidence():
    with pytest.raises(TraceError):
        Claim(id="a", text="t", confidence=1.5)


def test_verifiable_kinds():
    assert _claim("a", kind="arithmetic").verifiable()
    assert _claim("a", kind="code").verifiable()
    assert not _claim("a", kind="common").verifiable()


# ------------------------------------------------------------------- traces
def test_validate_accepts_dag():
    t = Trace(
        id="t",
        steps=[
            Step(id="s0", text="first", claims=[_claim("c1")]),
            Step(id="s1", text="second", claims=[_claim("c2", deps=("c1",))]),
        ],
        answer="1",
    )
    t.validate()
    assert t.is_valid()


def test_validate_rejects_cycle():
    t = Trace(
        id="t",
        steps=[
            Step(id="s0", text="a", claims=[_claim("c1", deps=("c2",))]),
            Step(id="s1", text="b", claims=[_claim("c2", deps=("c1",))]),
        ],
    )
    with pytest.raises(TraceError):
        t.validate()
    assert not t.is_valid()


def test_validate_rejects_dangling_dependency():
    t = Trace(id="t", steps=[Step(id="s", text="a", claims=[_claim("c1", deps=("ghost",))])])
    with pytest.raises(TraceError):
        t.validate()


def test_validate_rejects_duplicate_ids():
    t = Trace(
        id="t",
        steps=[
            Step(id="s0", text="a", claims=[_claim("c1")]),
            Step(id="s1", text="b", claims=[_claim("c1")]),
        ],
    )
    with pytest.raises(TraceError):
        t.validate()


def test_topo_order_respects_dependencies():
    t = Trace(
        id="t",
        steps=[
            Step(id="s0", text="a", claims=[_claim("c1")]),
            Step(id="s1", text="b", claims=[_claim("c2", deps=("c1",))]),
            Step(id="s2", text="c", claims=[_claim("c3", deps=("c2",))]),
        ],
    )
    order = [c.id for c in t.topo_order()]
    assert order.index("c1") < order.index("c2") < order.index("c3")


def test_depth_counts_longest_chain():
    t = Trace(
        id="t",
        steps=[
            Step(id="s0", text="a", claims=[_claim("c1")]),
            Step(id="s1", text="b", claims=[_claim("c2", deps=("c1",))]),
        ],
    )
    assert t.depth() == 2


def test_answer_claim_must_exist():
    t = Trace(id="t", steps=[Step(id="s", text="a", claims=[_claim("c1")])], answer_claim="nope")
    with pytest.raises(TraceError):
        t.validate()


# --------------------------------------------------------- answer normalising
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("60", "60"),
        ("60.0", "60"),
        (" 60 ", "60"),
        ("$60", "60"),
        ("60.00", "60"),
        ("3/4", "0.75"),
        ("Yes", "yes"),
        ("YES.", "yes"),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_normalize_answer_is_idempotent():
    for s in ["60", "$60.0", "Yes.", " 3/4 ", "hello world"]:
        once = normalize_answer(s)
        assert normalize_answer(once) == once


def test_answer_similarity():
    assert answer_similarity("60", "60") == 1.0
    assert answer_similarity("60", "60.0") == 1.0
    assert answer_similarity("60", "61") < 1.0
    assert answer_similarity("", "x") == 0.0
    assert 0.0 <= answer_similarity("abc", "abd") <= 1.0


def test_voting_key_collapses_equivalents():
    a = Trace(id="a", answer="60")
    b = Trace(id="b", answer=" 60.0 ")
    assert a.key() == b.key()
