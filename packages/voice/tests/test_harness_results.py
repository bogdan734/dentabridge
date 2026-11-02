import json
from datetime import date

import pytest

pytest.importorskip("sacrebleu")
from dentabridge_voice.harness.results import VoiceResults, render_markdown
from dentabridge_voice.harness.run import HarnessReport

REPORT = HarnessReport(
    n=50,
    n_silent=1,
    stage_ms={"stt": 812.4, "translate": 431.0, "tts_first": 120.6},
    p50_ms=1350.2,
    p95_ms=2210.9,
    wer=0.0412,
    chrf=71.34,
    glossary_acc=0.9,
    digit_acc=1.0,
)
RESULTS = VoiceResults(
    profile="mac-native",
    recorded_on=date(2026, 11, 2),
    hardware="Apple M3 Max, 36 GB",
    load_average=5.3,
    audio="synthetic speech from Piper voices, not human recordings",
    models={"stt": "faster-whisper small (int8, cpu)", "translator": "qwen3:4b-instruct"},
    report=REPORT,
)


def test_the_markdown_numbers_are_the_json_numbers() -> None:
    document = json.loads(RESULTS.model_dump_json())
    text = render_markdown(RESULTS)
    report = document["report"]
    assert f"{report['p50_ms'] / 1000:.2f} s" in text
    assert f"{report['p95_ms'] / 1000:.2f} s" in text
    assert f"{report['wer'] * 100:.1f} %" in text
    assert f"{report['chrf']:.1f}" in text
    assert f"{report['glossary_acc'] * 100:.0f} %" in text
    assert f"{report['digit_acc'] * 100:.0f} %" in text
    for stage, ms in report["stage_ms"].items():
        assert f"| {stage} | {ms:.0f} ms |" in text


def test_the_markdown_says_what_the_numbers_are_not() -> None:
    text = render_markdown(RESULTS)
    assert "synthetic speech" in text
    assert "not human recordings" in text
    assert "Apple M3 Max" in text
    assert "load average" in text


def test_a_run_with_no_audio_renders_dashes_not_crashes() -> None:
    empty = RESULTS.model_copy(
        update={
            "report": REPORT.model_copy(
                update={"p50_ms": None, "p95_ms": None, "glossary_acc": None}
            )
        }
    )
    assert "—" in render_markdown(empty)
