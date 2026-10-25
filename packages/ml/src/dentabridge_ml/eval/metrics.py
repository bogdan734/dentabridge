"""Per-finding sensitivity and specificity with cluster-bootstrap intervals.

Photos of one mouth are not independent, so the bootstrap resamples whole groups, never single
images. Sensitivity needs positives and specificity needs negatives: when a side is empty the
number is None ("not estimable"), never zero and never one.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dentabridge_core import FindingCode

REPS = 2000
ALPHA = 0.05
MIN_VALID_FRACTION = 0.9  # resamples that must contain the side being estimated


@dataclass(frozen=True)
class BinaryStats:
    code: FindingCode
    tp: int
    fp: int
    tn: int
    fn: int
    sens: float | None
    spec: float | None
    sens_ci: tuple[float, float] | None
    spec_ci: tuple[float, float] | None


def _ratio(hit: NDArray[np.float64], miss: NDArray[np.float64]) -> NDArray[np.float64]:
    total = hit + miss
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, hit / np.where(total > 0, total, 1), np.nan)


def _interval(values: NDArray[np.float64]) -> tuple[float, float] | None:
    valid = values[~np.isnan(values)]
    if len(valid) < MIN_VALID_FRACTION * len(values):
        return None
    low, high = np.percentile(valid, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def binary_stats(
    preds: Sequence[bool],
    labels: Sequence[bool],
    code: FindingCode,
    groups: Sequence[str],
    seed: int,
    reps: int = REPS,
) -> BinaryStats:
    if not (len(preds) == len(labels) == len(groups)):
        raise ValueError("preds, labels and groups must line up")
    p, y = np.asarray(preds, dtype=bool), np.asarray(labels, dtype=bool)
    tp, fp = int(np.sum(p & y)), int(np.sum(p & ~y))
    tn, fn = int(np.sum(~p & ~y)), int(np.sum(~p & y))
    names = sorted(set(groups))
    index = {name: i for i, name in enumerate(names)}
    per_group = np.zeros((len(names), 4))  # tp, fn, tn, fp
    for pred, label, name in zip(p, y, groups, strict=True):
        per_group[index[name], (0 if pred else 1) if label else (3 if pred else 2)] += 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(reps, len(names)))
    sums = per_group[draws].sum(axis=1)  # (reps, 4)
    sens_boot = _ratio(sums[:, 0], sums[:, 1])
    spec_boot = _ratio(sums[:, 2], sums[:, 3])
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return BinaryStats(
        code=code,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        sens=sens,
        spec=spec,
        sens_ci=_interval(sens_boot) if sens is not None else None,
        spec_ci=_interval(spec_boot) if spec is not None else None,
    )
