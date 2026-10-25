import pytest
from pydantic import ValidationError

from dentabridge_core import ASSESSABLE, Finding, FindingCode, Region


def test_assessable_sets() -> None:
    assert len(FindingCode) == 12
    assert ASSESSABLE["intraoral_photo"] == {
        FindingCode.CARIES,
        FindingCode.FILLING,
        FindingCode.CROWN,
        FindingCode.ABRASION,
        FindingCode.GINGIVITIS,
    }
    assert ASSESSABLE["panoramic_xray"] == {
        FindingCode.CARIES,
        FindingCode.PERIAPICAL_LESION,
        FindingCode.IMPACTED_TOOTH,
    }


@pytest.mark.parametrize(
    "box",
    [(0.5, 0.1, 0.4, 0.9), (0.1, 0.5, 0.9, 0.5), (-0.1, 0.0, 0.5, 0.5), (0.0, 0.0, 1.2, 1.0)],
)
def test_region_bounds(box: tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = box
    with pytest.raises(ValidationError):
        Region.model_validate({"x0": x0, "y0": y0, "x1": x1, "y1": y1})


def test_not_assessed_has_no_score() -> None:
    with pytest.raises(ValidationError, match="no score"):
        Finding(code=FindingCode.CARIES, status="not_assessed", score=0.4)
    assert Finding(code=FindingCode.CARIES, status="not_assessed").score is None


@pytest.mark.parametrize("status", ["not_seen", "not_assessed"])
def test_regions_only_on_possible_findings(status: str) -> None:
    region = {"x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2}
    with pytest.raises(ValidationError, match="only a possible"):
        Finding.model_validate({"code": "caries", "status": status, "regions": [region]})


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_score_range(score: float) -> None:
    with pytest.raises(ValidationError):
        Finding(code=FindingCode.CARIES, status="possible", score=score)


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Finding.model_validate({"code": "caries", "status": "possible", "diagnosis": "caries"})
