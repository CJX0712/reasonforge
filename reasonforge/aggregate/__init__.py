"""L5/L8 — aggregation and calibration."""

from .vwsc import (
    VWSCConfig,
    TraceScore,
    UNKNOWN_TRUST,
    majority_vote,
    score_trace,
    score_traces,
    vwsc_aggregate,
    weighted_vote,
)
from .calibration import (
    PlattCalibrator,
    calibration_error,
    fit_platt,
    fit_temperature,
    reliability_diagram,
)

__all__ = [
    "VWSCConfig",
    "TraceScore",
    "UNKNOWN_TRUST",
    "majority_vote",
    "score_trace",
    "score_traces",
    "vwsc_aggregate",
    "weighted_vote",
    "PlattCalibrator",
    "calibration_error",
    "fit_platt",
    "fit_temperature",
    "reliability_diagram",
]
