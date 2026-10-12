"""L3/L6 — inference-time search strategies."""

from .base import SolveContext, Strategy, StrategyResult, verify_all
from .self_consistency import (
    BestOfNVerified,
    Greedy,
    SelfConsistency,
    VWSCAdaptive,
    VWSCStrategy,
)
from .dvts import DVTS, DiverseVerifierTreeSearch
from .mcts import ARMS, UCBPolicyBandit

__all__ = [
    "SolveContext",
    "Strategy",
    "StrategyResult",
    "verify_all",
    "Greedy",
    "SelfConsistency",
    "BestOfNVerified",
    "VWSCStrategy",
    "VWSCAdaptive",
    "DVTS",
    "DiverseVerifierTreeSearch",
    "UCBPolicyBandit",
    "ARMS",
    "STRATEGIES",
]

STRATEGIES = {
    "greedy": Greedy,
    "self_consistency": SelfConsistency,
    "best_of_n_verified": BestOfNVerified,
    "vwsc": VWSCStrategy,
    "vwsc_adaptive": VWSCAdaptive,
    "dvts": DiverseVerifierTreeSearch,
    "ucb_policy_bandit": UCBPolicyBandit,
}


def make_strategy(name: str, **kwargs):
    """Instantiate a strategy by name."""
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}")
    cls = STRATEGIES[name]
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()
