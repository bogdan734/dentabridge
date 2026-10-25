from pathlib import Path

import pytest

from dentabridge_ml.pilot.roundtrip import ImportPath, RoundtripReport, serving_decision


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
