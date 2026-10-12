"""
L4 invariants — the verification layer.

A verifier must be *sound*: when it says False, the claim is genuinely wrong.
False positives (calling a correct claim wrong) would silently poison the whole
system, so the bulk of these tests pin down both directions.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import pytest

from reasonforge.core.trace import Claim, Step, Trace
from reasonforge.verifiers import (
    AlgebraVerifier,
    ArithmeticVerifier,
    CodeVerifier,
    ConsistencyVerifier,
    ConstraintVerifier,
    LogicVerifier,
    default_registry,
)
from reasonforge.verifiers.logic import (
    LogicError,
    dpll,
    entails,
    find_model,
    parse_formula,
    satisfiable,
    to_cnf,
    valid,
)
from reasonforge.verifiers.constraint import CSP, ac3, check_assignment, solve_csp


# --------------------------------------------------------------- arithmetic
@pytest.mark.parametrize(
    "expr,want",
    [
        ("12 * 5 = 60", True),
        ("12 * 5 = 61", False),
        ("15% of 200 = 30", True),
        ("480 / 40 = 12", True),
        ("2 ** 10 = 1024", True),
        ("1 / 3 = 0.33", False),
        ("7 + 8", None),          # bare expression: no assertion to check
        ("(((", None),            # unparseable
        ("1 / 0 = 1", None),      # division by zero -> unknown, never a crash
        ("9 ** 9 ** 9 = 1", None),  # unsafe power -> unknown, never a hang
    ],
)
def test_arithmetic(expr, want):
    r = ArithmeticVerifier().verify(Claim(id="c", text="t", kind="arithmetic", expr=expr))
    assert r.ok is want


def test_arithmetic_attaches_counterexample():
    r = ArithmeticVerifier().verify(
        Claim(id="c", text="t", kind="arithmetic", expr="12 * 5 = 61")
    )
    assert r.failed
    assert r.counterexample


def test_arithmetic_never_uses_eval():
    src = "__import__('os').system('echo pwned')"
    r = ArithmeticVerifier().verify(Claim(id="c", text="t", kind="arithmetic", expr=src))
    assert r.ok is None


# ------------------------------------------------------------------- logic
@pytest.mark.parametrize(
    "expr,want",
    [
        ("rain, rain -> wet |= wet", True),          # modus ponens
        ("wet, rain -> wet |= rain", False),         # affirming the consequent
        ("A, A -> B, B -> C |= C", True),            # hypothetical syllogism
        ("A -> B, not B |= not A", True),            # modus tollens
        ("A or not A", True),                        # excluded middle
        ("A and not A", False),                      # contradiction
        ("not (A and B) <-> (not A or not B)", True),  # De Morgan
        ("(A -> B) <-> (B -> A)", False),            # implication is not symmetric
        ("SAT: wet, rain -> wet, not rain", True),   # countermodel exists
        ("UNSAT: A and not A", True),
        ("VALID: A -> A", True),
        ("INVALID: A -> B", True),
        ("VALID: A -> B", False),
    ],
)
def test_logic(expr, want):
    r = LogicVerifier().verify(Claim(id="c", text="t", kind="logic", expr=expr))
    assert r.ok is want, f"{expr}: {r.detail}"


def test_dpll_model_satisfies_every_clause():
    clauses = [(1, -2), (2, 3), (-1, 3), (-3, 1)]
    model = dpll(clauses)
    assert model is not None
    for c in clauses:
        assert any((lit > 0 and model.get(abs(lit))) or (lit < 0 and not model.get(abs(lit))) for lit in c)


def test_dpll_detects_unsat():
    assert dpll([(1,), (-1,)]) is None


def test_tseitin_preserves_satisfiability():
    for src in ["A and B", "A or not B", "A -> B", "not (A and B) <-> (not A or not B)"]:
        node = parse_formula(src)
        clauses, _ = to_cnf(node)
        assert satisfiable(node) == (dpll(clauses) is not None)


def test_entails_and_valid_agree():
    assert entails(["A", "A -> B"], "B")
    assert valid("A -> A")
    assert not valid("A")
    assert find_model("A and not A") is None
    assert find_model("A") is not None


def test_logic_rejects_malformed():
    with pytest.raises(LogicError):
        parse_formula("A and")
    r = LogicVerifier().verify(Claim(id="c", text="t", kind="logic", expr="A and and B"))
    assert r.ok is None


# -------------------------------------------------------------- constraints
def test_constraint_assignment_ok():
    r = ConstraintVerifier().verify(
        Claim(id="c", text="t", kind="constraint",
              expr="x=6; y=4 ### x + y = 10, 2*x + 4*y = 28")
    )
    assert r.passed


def test_constraint_assignment_violated():
    r = ConstraintVerifier().verify(
        Claim(id="c", text="t", kind="constraint",
              expr="x=6; y=5 ### x + y = 10, 2*x + 4*y = 28")
    )
    assert r.failed
    assert r.counterexample


def test_csp_solver_finds_known_model():
    csp = CSP({"x": [float(i) for i in range(0, 11)],
               "y": [float(i) for i in range(0, 11)]},
              ["x + y = 10", "2*x + 4*y = 28"])
    model = solve_csp(csp)
    assert model is not None
    assert check_assignment(model, ["x + y = 10", "2*x + 4*y = 28"])[0]


def test_ac3_prunes_impossible_values():
    csp = CSP({"x": [1.0, 2.0, 3.0]}, ["x > 5"])
    assert not ac3(csp)


# --------------------------------------------------------------------- code
def test_code_executes_and_compares():
    code = "s = 0\nfor i in range(1, 101):\n    s = s + i\nresult = s"
    r = CodeVerifier().verify(Claim(id="c", text="t", kind="code", expr=code + "###5050"))
    assert r.passed, r.detail


def test_code_detects_wrong_result():
    code = "s = 0\nfor i in range(1, 100):\n    s = s + i\nresult = s"
    r = CodeVerifier().verify(Claim(id="c", text="t", kind="code", expr=code + "###5050"))
    assert r.failed
    assert r.counterexample


def test_sandbox_blocks_imports():
    r = CodeVerifier().verify(
        Claim(id="c", text="t", kind="code", expr="import os\nresult = 1###1")
    )
    assert r.ok is None


def test_sandbox_blocks_dunder_and_attributes():
    r = CodeVerifier().verify(
        Claim(id="c", text="t", kind="code", expr="x = (1).__class__\nresult = x###1")
    )
    assert r.ok is None


def test_sandbox_caps_infinite_loops():
    r = CodeVerifier().verify(
        Claim(id="c", text="t", kind="code", expr="while True:\n    pass\nresult = 1###1")
    )
    assert r.ok is None


def test_sandbox_supports_functions_and_recursion_depth_cap():
    code = "def f(n):\n    if n <= 1:\n        return 1\n    return n * f(n - 1)\nresult = f(5)"
    r = CodeVerifier().verify(Claim(id="c", text="t", kind="code", expr=code + "###120"))
    assert r.passed, r.detail


# -------------------------------------------------------------- consistency
def test_consistency_matches():
    t = Trace(id="t", answer="68")
    r = ConsistencyVerifier().verify(
        Claim(id="c", text="t", kind="consistency", expr="68"), trace=t
    )
    assert r.passed


def test_consistency_catches_drift():
    t = Trace(id="t", answer="70")
    r = ConsistencyVerifier().verify(
        Claim(id="c", text="t", kind="consistency", expr="68"), trace=t
    )
    assert r.failed
    assert "70" in (r.counterexample or "")


def test_consistency_needs_trace():
    r = ConsistencyVerifier().verify(Claim(id="c", text="t", kind="consistency", expr="68"))
    assert r.ok is None


# --------------------------------------------------------------- the registry
def test_registry_routes_by_kind():
    reg = default_registry()
    assert set(reg.kinds()) == {"arithmetic", "algebra", "logic", "constraint", "code", "consistency"}


def test_registry_takes_strictest_verdict():
    class AlwaysFalse(ArithmeticVerifier):
        name = "always-false"

        def verify(self, claim, trace=None):
            return self._result(claim, False, 1.0, "hard refutation")

    reg = default_registry([AlwaysFalse()])
    r = reg.verify_claim(Claim(id="c", text="t", kind="arithmetic", expr="1 = 1"))
    assert r.failed  # one refutation beats any number of passes


def test_registry_reports_unknown_for_unknown_kind():
    reg = default_registry()
    r = reg.verify_claim(Claim(id="c", text="t", kind="common", expr="sky is blue"))
    assert r.ok is None


def test_report_counts():
    reg = default_registry()
    trace = Trace(
        id="t",
        steps=[
            Step(id="s0", text="a", claims=[Claim(id="c1", text="a", kind="arithmetic", expr="1 = 1")]),
            Step(id="s1", text="b", claims=[Claim(id="c2", text="b", kind="arithmetic", expr="1 = 2")]),
        ],
        answer="1",
    )
    rep = reg.verify_trace(trace)
    assert rep.n_passed() == 1
    assert rep.n_failed() == 1
    assert rep.counterexamples()


# ------------------------------------------------------------------- algebra
def test_algebra_identity():
    r = AlgebraVerifier().verify(
        Claim(id="c", text="t", kind="algebra", expr="(x+1)**2 = x**2 + 2*x + 1")
    )
    assert r.passed, r.detail


def test_algebra_rejects_false_identity():
    r = AlgebraVerifier().verify(
        Claim(id="c", text="t", kind="algebra", expr="(x+1)**2 = x**2 + 2*x + 2")
    )
    assert r.failed, r.detail
