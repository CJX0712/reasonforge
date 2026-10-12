"""
L8 — Confidence calibration (置信度校准).

A confidence number that cannot be trusted is worse than no confidence number.
This module measures and fixes calibration for the answer-sharing confidence
produced by VWSC.

* :func:`reliability_diagram` — bin-level accuracy vs confidence
* :func:`expected_calibration_error` — re-exported from L0
* :func:`fit_platt` — Platt scaling (logistic calibration) fitted by Newton steps
* :func:`fit_temperature` — single-parameter temperature scaling

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..core.numeric import DeterministicRNG, expected_calibration_error

__all__ = [
    "reliability_diagram",
    "calibration_error",
    "fit_platt",
    "fit_temperature",
    "PlattCalibrator",
]


def reliability_diagram(
    confidences: Sequence[float], corrects: Sequence[int], n_bins: int = 10
) -> List[Dict[str, float]]:
    """Per-bin ``(confidence, accuracy, count)`` rows."""
    if len(confidences) != len(corrects):
        raise ValueError("length mismatch")
    bins_acc = [0.0] * n_bins
    bins_conf = [0.0] * n_bins
    bins_n = [0] * n_bins
    for c, y in zip(confidences, corrects):
        c = min(1.0, max(0.0, float(c)))
        b = min(n_bins - 1, int(c * n_bins))
        bins_acc[b] += float(y)
        bins_conf[b] += c
        bins_n[b] += 1
    out: List[Dict[str, float]] = []
    for b in range(n_bins):
        if bins_n[b] == 0:
            continue
        out.append(
            {
                "bin": b,
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "mean_confidence": bins_conf[b] / bins_n[b],
                "accuracy": bins_acc[b] / bins_n[b],
                "count": float(bins_n[b]),
            }
        )
    return out


def calibration_error(
    confidences: Sequence[float], corrects: Sequence[int], n_bins: int = 10
) -> Dict[str, float]:
    """ECE plus the max per-bin gap (MCE) and overall accuracy."""
    ece = expected_calibration_error(confidences, corrects, n_bins)
    rows = reliability_diagram(confidences, corrects, n_bins)
    mce = max((abs(r["accuracy"] - r["mean_confidence"]) for r in rows), default=0.0)
    acc = (sum(corrects) / len(corrects)) if corrects else 0.0
    return {"ece": ece, "mce": mce, "accuracy": acc, "n": float(len(corrects))}


def fit_temperature(
    logits: Sequence[float], labels: Sequence[int], iters: int = 200, lr: float = 0.05
) -> float:
    """Single-parameter temperature scaling by gradient descent on NLL.

    ``logits`` are *pre-sigmoid* scores; ``labels`` in {0,1}.
    """
    if len(logits) != len(labels) or not logits:
        return 1.0
    T = 1.0
    for _ in range(iters):
        g = 0.0
        for z, y in zip(logits, labels):
            p = 1.0 / (1.0 + math.exp(-(z / T)))
            g += (p - y) * (z / (T * T))
        g /= len(logits)
        T -= lr * g
        T = max(0.05, min(50.0, T))
    return T


@dataclass
class PlattCalibrator:
    """Platt scaling: ``P(y=1) = sigmoid(A·s + B)`` fitted by Newton-IRLS."""

    A: float = 1.0
    B: float = 0.0

    def predict(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-(self.A * score + self.B)))

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        iters: int = 50,
        rng: DeterministicRNG = DeterministicRNG(0),
    ) -> "PlattCalibrator":
        """Newton steps on the regularised logistic loss (Lin et al. 2007)."""
        if len(scores) != len(labels) or not scores:
            return self
        n = len(scores)
        prior_pos = max(1e-6, sum(labels) / n)
        prior_neg = 1.0 - prior_pos
        self.B = math.log(prior_pos / prior_neg)
        self.A = 1.0
        for _ in range(iters):
            gA = gB = 0.0
            hAA = hBB = hAB = 0.0
            for s, y in zip(scores, labels):
                z = self.A * s + self.B
                p = 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0
                d = p - y
                gA += d * s
                gB += d
                w = max(1e-9, p * (1.0 - p))
                hAA += w * s * s
                hBB += w
                hAB += w * s
            # Levenberg-style damping for stability
            hAA += 1e-6
            hBB += 1e-6
            det = hAA * hBB - hAB * hAB
            if abs(det) < 1e-12:
                break
            dA = -(hBB * gA - hAB * gB) / det
            dB = -(-hAB * gA + hAA * gB) / det
            self.A += dA
            self.B += dB
            if abs(dA) < 1e-8 and abs(dB) < 1e-8:
                break
        return self

    def as_dict(self) -> Dict[str, float]:
        return {"A": self.A, "B": self.B}


def fit_platt(scores: Sequence[float], labels: Sequence[int]) -> PlattCalibrator:
    """Convenience wrapper returning a fitted :class:`PlattCalibrator`."""
    return PlattCalibrator().fit(scores, labels)
