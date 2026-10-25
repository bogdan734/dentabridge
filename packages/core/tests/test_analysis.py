from datetime import timedelta

import pytest
from pydantic import ValidationError

from dentabridge_core import AnalysisResult, Finding, FindingCode
from dentabridge_core.testing import sample_analysis


def _with(**changes: object) -> AnalysisResult:
    data = sample_analysis().model_dump()
    data.update(changes)
    return AnalysisResult.model_validate(data)


def _findings() -> list[dict[str, object]]:
    return [f.model_dump() for f in sample_analysis().findings]


def test_sample_analysis_is_valid_and_complete() -> None:
    result = sample_analysis({FindingCode.CARIES})
    assert {f.code for f in result.findings} == set(FindingCode)
    assert next(f for f in result.findings if f.code == FindingCode.CARIES).status == "possible"
    assert next(f for f in result.findings if f.code == FindingCode.CALCULUS).status == (
        "not_assessed"
    )


def test_analysis_json_roundtrip() -> None:
    original = sample_analysis({FindingCode.CARIES, FindingCode.GINGIVITIS}, locale="ru")
    assert AnalysisResult.model_validate_json(original.model_dump_json()) == original


def test_analysis_requires_every_catalog_code() -> None:
    findings = _findings()
    with pytest.raises(ValidationError, match="every catalogue code"):
        _with(findings=findings[:-1])
    with pytest.raises(ValidationError, match="every catalogue code"):
        _with(findings=[*findings[:-1], findings[0]])


def test_non_assessable_code_must_be_not_assessed() -> None:
    findings = _findings()
    for finding in findings:
        if finding["code"] == FindingCode.PERIAPICAL_LESION:
            finding.update(status="possible", score=0.5)
    with pytest.raises(ValidationError, match="cannot be assessed"):
        _with(findings=findings)


def test_unusable_image_needs_a_reason_and_no_findings() -> None:
    nothing = [Finding(code=code, status="not_assessed").model_dump() for code in FindingCode]
    assert _with(image_quality="unusable", quality_reason="blurry", findings=nothing).urgency
    with pytest.raises(ValidationError, match="quality_reason"):
        _with(image_quality="unusable", findings=nothing)
    with pytest.raises(ValidationError, match="quality_reason"):
        _with(quality_reason="blurry")
    with pytest.raises(ValidationError, match="unusable image"):
        _with(image_quality="unusable", quality_reason="blurry")


@pytest.mark.parametrize(
    ("note", "status"),
    [("a note", "absent"), (None, "passed"), ("a note", "rejected")],
)
def test_note_is_present_exactly_when_passed(note: str | None, status: str) -> None:
    with pytest.raises(ValidationError, match="vlm_note"):
        _with(vlm_note=note, vlm_note_status=status)
    assert _with(vlm_note="a note", vlm_note_status="passed").vlm_note == "a note"


def test_expiry_is_after_creation() -> None:
    created = sample_analysis().created_at
    with pytest.raises(ValidationError, match="expires_at"):
        _with(expires_at=created)
    assert _with(expires_at=created + timedelta(seconds=1)).expires_at > created


def test_naive_datetimes_are_rejected() -> None:
    naive = sample_analysis().created_at.replace(tzinfo=None)
    with pytest.raises(ValidationError):
        _with(created_at=naive)
