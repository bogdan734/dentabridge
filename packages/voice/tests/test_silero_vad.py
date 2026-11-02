from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faster_whisper")
pytest.importorskip("soundfile")
import soundfile  # type: ignore[import-untyped]  # ships no type information

from dentabridge_core import ReplayManifest
from dentabridge_voice.adapters.silero_vad import RATE, WINDOW, SileroVad
from dentabridge_voice.harness.audio import to_pcm16

DEMO = Path(__file__).resolve().parents[1] / "replays" / "demo"


def probabilities(samples: np.ndarray, start_ms: int, end_ms: int) -> list[float]:
    vad = SileroVad()
    piece = samples[start_ms * RATE // 1000 : end_ms * RATE // 1000]
    windows = [piece[i : i + WINDOW] for i in range(0, len(piece) - WINDOW + 1, WINDOW)]
    return [vad.probability(to_pcm16(w)) for w in windows]


def test_speech_is_told_from_silence() -> None:
    """The patient's track of the demo replay: their own words in their turn, silence in the
    dentist's (the replay holds synthetic speech, so this is a sanity check, not a benchmark)."""
    manifest = ReplayManifest.model_validate_json((DEMO / "manifest.json").read_text())
    samples, rate = soundfile.read(DEMO / "audio" / "patient-ru.ogg", dtype="float32")
    assert rate == RATE
    patient_turn = next(e for e in manifest.events if e.speaker == "patient")
    dentist_turn = next(e for e in manifest.events if e.speaker == "dentist")
    end = patient_turn.t_speech_end_ms
    assert max(probabilities(samples, end - 1500, end - 200)) > 0.8
    end = dentist_turn.t_speech_end_ms
    assert max(probabilities(samples, end - 1500, end - 200)) < 0.2


def test_a_window_of_the_wrong_size_is_refused() -> None:
    with pytest.raises(ValueError, match="1024 bytes"):
        SileroVad().probability(bytes(1000))
