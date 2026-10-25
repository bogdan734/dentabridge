import json

import pytest

from dentabridge_core import FindingCode
from dentabridge_ml.serving.parse import ParseError, ReleaseConfig, parse_output

C, F = FindingCode.CARIES, FindingCode.FILLING
BOX = {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.6}
RELEASE = ReleaseConfig(assessed=frozenset({C, F}))


def _answer(*findings: dict[str, object], quality: str = "ok", reason: str | None = None) -> str:
    return json.dumps({"image_quality": quality, "quality_reason": reason, "findings": findings})


VALID = [
    _answer({"code": "caries", "status": "possible", "regions": [BOX]}),
    _answer({"code": "caries", "status": "not_seen", "regions": []}),
    _answer(
        {"code": "caries", "status": "possible", "regions": [BOX]},
        {"code": "filling", "status": "not_seen", "regions": []},
    ),
    "```json\n" + _answer({"code": "filling", "status": "not_seen"}) + "\n```",
    '{"image_quality": "ok", "findings": [{"code": "filling", "status": "possible"}]}',
    _answer(quality="unusable", reason="blurry"),
    "  " + _answer({"code": "caries", "status": "not_seen"}) + "\n",
]
INVALID = [
    "I think there is a cavity on the left.",
    '{"image_quality": "ok", "findings": [',
    _answer({"code": "caries", "status": "maybe"}),
    _answer({"code": "wisdom_tooth", "status": "possible"}),
    _answer({"code": "caries", "status": "not_assessed"}),
    _answer(quality="unusable"),
    _answer(
        {
            "code": "caries",
            "status": "possible",
            "regions": [{"x0": 0.6, "y0": 0.1, "x1": 0.2, "y1": 0.4}],
        }
    ),
    _answer({"code": "caries", "status": "not_seen"}, {"code": "caries", "status": "possible"}),
]


@pytest.mark.parametrize("raw", VALID)
def test_parse_golden_valid(raw: str) -> None:
    output = parse_output(raw, None, RELEASE)
    assert output.image_quality in {"ok", "unusable"}


@pytest.mark.parametrize("raw", INVALID)
def test_parse_golden_invalid_is_an_error_not_a_guess(raw: str) -> None:
    with pytest.raises(ParseError):
        parse_output(raw, None, RELEASE)


def test_release_marks_unshipped_codes_not_assessed() -> None:
    raw = _answer(
        {"code": "caries", "status": "possible", "regions": [BOX]},
        {"code": "gingivitis", "status": "possible", "regions": [BOX]},
    )
    findings = {f.code: f for f in parse_output(raw, None, RELEASE).findings}
    assert findings[C].status == "possible"
    assert findings[C].regions[0].x1 == 0.5
    assert findings[FindingCode.GINGIVITIS].status == "not_assessed"
    assert findings[FindingCode.GINGIVITIS].regions == ()


def test_threshold_overrides_the_status_with_the_score() -> None:
    release = ReleaseConfig(assessed=frozenset({C}), thresholds={C: 0.6}, needs_scores=True)
    possible = _answer({"code": "caries", "status": "possible", "regions": [BOX]})
    low = parse_output(possible, {C: 0.4}, release).findings[0]
    high = parse_output(possible, {C: 0.8}, release).findings[0]
    assert (low.status, low.score, low.regions) == ("not_seen", 0.4, ())
    assert (high.status, high.score) == ("possible", 0.8)
    assert len(high.regions) == 1


def test_a_release_that_needs_scores_refuses_a_backend_without_them() -> None:
    release = ReleaseConfig(assessed=frozenset({C}), needs_scores=True)
    with pytest.raises(ParseError, match="token scores"):
        parse_output(_answer({"code": "caries", "status": "not_seen"}), None, release)


def test_release_file_roundtrip(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "release.json"
    release = ReleaseConfig(assessed=frozenset({C}), thresholds={C: 0.5}, needs_scores=True)
    path.write_text(release.model_dump_json())
    assert ReleaseConfig.read(path) == release
