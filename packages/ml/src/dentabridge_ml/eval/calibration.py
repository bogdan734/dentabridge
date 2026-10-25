"""Calibration and selective prediction: is a confident answer actually more often right?"""

import numpy as np
from numpy.typing import NDArray


def ece(scores: NDArray[np.float64], labels: NDArray[np.bool_], bins: int = 10) -> float:
    """Expected calibration error of P(finding present) against the true labels."""
    if len(scores) != len(labels) or len(scores) == 0:
        raise ValueError("scores and labels must be the same non-zero length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return total


def risk_coverage(
    conf: NDArray[np.float64], correct: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Answer the most confident first: error rate at each coverage, and the area under it."""
    if len(conf) != len(correct) or len(conf) == 0:
        raise ValueError("conf and correct must be the same non-zero length")
    order = np.argsort(-conf, kind="stable")
    errors = (~correct[order]).astype(np.float64)
    answered = np.arange(1, len(conf) + 1, dtype=np.float64)
    risk = np.cumsum(errors) / answered
    coverage = answered / len(conf)
    return coverage, risk, float(risk.mean())
