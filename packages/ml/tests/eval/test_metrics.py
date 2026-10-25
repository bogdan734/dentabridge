import numpy as np
import pytest

from dentabridge_core import FindingCode
from dentabridge_ml.eval.calibration import ece, risk_coverage
from dentabridge_ml.eval.metrics import binary_stats

C = FindingCode.CARIES


def test_binary_stats_known_counts() -> None:
    preds = [True, True, False, False, True, False, False, False]
    labels = [True, True, True, False, False, False, False, False]
    stats = binary_stats(preds, labels, C, [f"g{i}" for i in range(8)], seed=1)
    assert (stats.tp, stats.fn, stats.fp, stats.tn) == (2, 1, 1, 4)
    assert stats.sens == pytest.approx(2 / 3)
    assert stats.spec == pytest.approx(4 / 5)


def test_ci_deterministic_and_contains_point() -> None:
    rng = np.random.default_rng(0)
    labels = [bool(v) for v in rng.random(300) < 0.4]
    preds = [bool(v) if rng.random() < 0.85 else not v for v in labels]
    groups = [f"g{i // 3}" for i in range(300)]
    first = binary_stats(preds, labels, C, groups, seed=7)
    again = binary_stats(preds, labels, C, groups, seed=7)
    assert first == again
    assert first.sens_ci is not None
    assert first.sens_ci[0] <= first.sens <= first.sens_ci[1]  # type: ignore[operator]
    assert first.spec_ci is not None
    assert first.spec_ci[0] <= first.spec <= first.spec_ci[1]  # type: ignore[operator]


def _width(interval: tuple[float, float]) -> float:
    return interval[1] - interval[0]


def test_cluster_bootstrap_is_wider_than_per_image_when_photos_are_correlated() -> None:
    # 40 mouths x 5 photos, each mouth wholly right or wholly wrong: 5x fewer independent facts
    labels = [True] * 200
    preds = [(i // 5) % 2 == 0 for i in range(200)]
    clustered = binary_stats(preds, labels, C, [f"m{i // 5}" for i in range(200)], seed=3)
    per_image = binary_stats(preds, labels, C, [f"i{i}" for i in range(200)], seed=3)
    assert clustered.sens_ci is not None
    assert per_image.sens_ci is not None
    assert _width(clustered.sens_ci) > 1.8 * _width(per_image.sens_ci)


def test_a_side_without_examples_is_not_estimable() -> None:
    only_positive = binary_stats([True, False], [True, True], C, ["a", "b"], seed=0)
    assert (only_positive.spec, only_positive.spec_ci) == (None, None)
    assert only_positive.sens == 0.5
    only_negative = binary_stats([False, True], [False, False], C, ["a", "b"], seed=0)
    assert (only_negative.sens, only_negative.sens_ci) == (None, None)
    assert only_negative.spec == 0.5


def test_inputs_must_line_up() -> None:
    with pytest.raises(ValueError, match="line up"):
        binary_stats([True], [True, False], C, ["a"], seed=0)


def test_ece_zero_when_perfect_and_large_when_overconfident() -> None:
    labels = np.array([True, False, True, False])
    assert ece(np.array([1.0, 0.0, 1.0, 0.0]), labels) == 0.0
    sure = np.full(100, 0.9)
    half = np.array([True, False] * 50)
    assert ece(sure, half) == pytest.approx(0.4)


def test_risk_coverage_monotone_for_a_good_ranking_and_worse_for_a_random_one() -> None:
    n = 200
    correct = np.array([True] * 150 + [False] * 50)
    good_conf = np.linspace(1.0, 0.0, n)  # the 150 right answers are the most confident
    coverage, risk, aurc = risk_coverage(good_conf, correct)
    assert np.all(np.diff(coverage) > 0)
    assert np.all(np.diff(risk) >= -1e-12)
    shuffled = np.random.default_rng(0).permutation(n).astype(float)
    assert aurc < risk_coverage(shuffled, correct)[2]
    assert risk[-1] == pytest.approx(0.25)


def test_calibration_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="same non-zero length"):
        ece(np.array([0.5]), np.array([True, False]))
    with pytest.raises(ValueError, match="same non-zero length"):
        risk_coverage(np.array([]), np.array([], dtype=bool))
