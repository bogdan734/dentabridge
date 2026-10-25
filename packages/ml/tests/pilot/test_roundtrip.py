import io
from pathlib import Path

import pytest
from PIL import Image

from dentabridge_core import Finding, FindingCode, VisionOutput
from dentabridge_ml.pilot.roundtrip import (
    COLOURS,
    ImportPath,
    RoundtripReport,
    agreement,
    colour_probe,
    same_answer,
    serving_decision,
    valid_rate,
)


def _report(
    path: ImportPath = "safetensors",
    *,
    vision_ok: bool = True,
    json_valid_rate: float = 1.0,
    agreement: float = 1.0,
) -> RoundtripReport:
    return RoundtripReport(
        path=path,
        vision_ok=vision_ok,
        json_valid_rate=json_valid_rate,
        agreement=agreement,
        logprobs_available=True,
        ollama_version="0.0.0",
    )


@pytest.mark.parametrize(
    ("reports", "expected"),
    [
        ([_report()], "ollama"),
        ([_report(agreement=0.9, json_valid_rate=0.95)], "ollama"),
        ([_report(agreement=0.89)], "mlx_or_transformers"),
        ([_report(json_valid_rate=0.94)], "mlx_or_transformers"),
        ([_report(vision_ok=False)], "mlx_or_transformers"),
        ([_report(vision_ok=False), _report("gguf_mmproj")], "ollama"),
        ([], "mlx_or_transformers"),
    ],
)
def test_decision_rule_table(reports: list[RoundtripReport], expected: str) -> None:
    assert serving_decision(reports) == expected


def test_report_roundtrips_json(tmp_path: Path) -> None:
    report = _report("gguf_mmproj", agreement=0.95)
    path = tmp_path / "roundtrip.json"
    report.write(path)
    assert RoundtripReport.model_validate_json(path.read_text()) == report


def _answer(quality: str = "ok", *statuses: str, boxed: bool = False) -> VisionOutput:
    codes = [FindingCode.CARIES, FindingCode.FILLING]
    findings = tuple(
        Finding(code=code, status=status)  # type: ignore[arg-type]
        for code, status in zip(codes, statuses, strict=False)
    )
    return VisionOutput(image_quality=quality, findings=findings, latency_ms=12.0 if boxed else 0.0)  # type: ignore[arg-type]


def test_same_answer_compares_quality_and_statuses_not_timing() -> None:
    assert same_answer(
        _answer("ok", "possible", "not_seen"), _answer("ok", "possible", "not_seen", boxed=True)
    )
    assert not same_answer(_answer("ok", "possible"), _answer("ok", "not_seen"))
    assert not same_answer(_answer("ok", "possible"), _answer("ok", "possible", "not_seen"))


def test_two_unparseable_answers_do_not_count_as_agreement() -> None:
    assert not same_answer(None, None)
    assert not same_answer(_answer("ok"), None)


def test_agreement_and_valid_rate() -> None:
    reference = [
        _answer("ok", "possible"),
        _answer("ok", "not_seen"),
        None,
        _answer("ok", "possible"),
    ]
    other = [_answer("ok", "possible"), _answer("ok", "possible"), None, None]
    assert agreement(reference, other) == 0.25
    assert valid_rate(other) == 0.5
    with pytest.raises(ValueError, match="same number"):
        agreement(reference, other[:2])


@pytest.mark.parametrize("name", sorted(COLOURS))
def test_colour_probe_is_a_big_square_of_that_colour(name: str) -> None:
    with Image.open(io.BytesIO(colour_probe(name))) as image:
        centre = image.convert("RGB").getpixel((image.width // 2, image.height // 2))
        corner = image.convert("RGB").getpixel((2, 2))
    assert all(abs(a - b) <= 25 for a, b in zip(centre, COLOURS[name], strict=True))  # type: ignore[arg-type]
    assert corner != centre
