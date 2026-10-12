"""
L4 — Verification layer package.

:func:`default_registry` wires every built-in verifier. Imports are defensive:
a missing optional dependency (sympy) degrades gracefully instead of breaking.

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from typing import Optional, Sequence

from .base import VerificationReport, VerificationResult, Verifier, VerifierRegistry
from .arithmetic import ArithmeticVerifier
from .algebra import AlgebraVerifier
from .logic import LogicVerifier
from .constraint import ConstraintVerifier
from .code import CodeVerifier
from .consistency import ConsistencyVerifier

__all__ = [
    "VerificationReport",
    "VerificationResult",
    "Verifier",
    "VerifierRegistry",
    "ArithmeticVerifier",
    "AlgebraVerifier",
    "LogicVerifier",
    "ConstraintVerifier",
    "CodeVerifier",
    "ConsistencyVerifier",
    "default_registry",
]


def default_registry(extra: Optional[Sequence[Verifier]] = None) -> VerifierRegistry:
    """Registry with all built-in verifiers (plus any user-supplied ones)."""
    reg = VerifierRegistry(
        [
            ArithmeticVerifier(),
            AlgebraVerifier(),
            LogicVerifier(),
            ConstraintVerifier(),
            CodeVerifier(),
            ConsistencyVerifier(),
        ]
    )
    for v in extra or ():
        reg.register(v)
    return reg
