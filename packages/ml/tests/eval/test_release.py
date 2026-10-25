from pathlib import Path

import pytest

from dentabridge_core import FindingCode
from dentabridge_ml.eval.results import CodeResult, SplitResult
from dentabridge_ml.release import Gate, apply_gate, passes

GATE_FILE = Path(__file__).resolve().parents[2] / "configs" / "release_gate.toml"
GATE = Gate.read(GATE_FILE)


def _code(**changes: object) -> CodeResult:
    base: dict[str, object] = {
        "code": FindingCode.CARIES,
        "n_pos": 120,
        "n_neg": 400,
        "sens": 0.8,
        "spec": 0.9,
        "sens_ci": (0.7, 0.88),
        "spec_ci": (0.85, 0.94),
        "ece": 0.05,
    }
    return CodeResult.model_validate({**base, **changes})


def test_gate_thresholds() -> None:
    assert (GATE.min_positives, GATE.min_sens_lower, GATE.min_spec_lower, GATE.max_ece) == (
        30,
        0.60,
        0.60,
        0.10,
    )
    assert passes(_code(), GATE)
    assert not passes(_code(n_pos=29), GATE)
    assert not passes(_code(sens_ci=(0.59, 0.9)), GATE)
    assert not passes(_code(spec_ci=(0.59, 0.9)), GATE)
    assert not passes(_code(ece=0.11), GATE)


def test_boundaries_are_inclusive() -> None:
    assert passes(_code(n_pos=30, sens_ci=(0.60, 0.9), spec_ci=(0.60, 0.9), ece=0.10), GATE)


@pytest.mark.parametrize("missing", ["sens_ci", "spec_ci", "ece"])
def test_a_missing_number_fails_the_gate(missing: str) -> None:
    assert not passes(_code(**{missing: None}), GATE)


def test_no_negatives_means_not_assessed_not_a_free_pass() -> None:
    gingivitis = _code(
        code=FindingCode.GINGIVITIS, n_neg=0, spec=None, spec_ci=None, sens_ci=(0.8, 0.95)
    )
    assert not passes(gingivitis, GATE)


def test_apply_gate_lists_only_the_passing_codes() -> None:
    result = SplitResult(
        model_id="m",
        split="test",
        n_images=500,
        json_valid_rate=0.99,
        codes=(_code(), _code(code=FindingCode.FILLING, ece=0.3), _code(code=FindingCode.CROWN)),
    )
    release = apply_gate(result, GATE)
    assert release.assessed == {FindingCode.CARIES, FindingCode.CROWN}


def test_results_roundtrip_through_json(tmp_path: Path) -> None:
    result = SplitResult(
        model_id="m", split="val", n_images=1, json_valid_rate=1.0, codes=(_code(),)
    )
    path = tmp_path / "val.json"
    result.write(path)
    assert SplitResult.read(path) == result
