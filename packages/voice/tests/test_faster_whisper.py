import asyncio
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("faster_whisper")
from dentabridge_voice.adapters import faster_whisper as adapter
from dentabridge_voice.ports import SpeechToText


class FakeModel:
    instances: ClassVar[list["FakeModel"]] = []

    def __init__(self, name: str, device: str, compute_type: str) -> None:
        self.args = (name, device, compute_type)
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []
        FakeModel.instances.append(self)

    def transcribe(self, audio: np.ndarray, **options: Any) -> tuple[list[Any], None]:
        self.calls.append((audio, options))
        return [SimpleNamespace(text=" my tooth"), SimpleNamespace(text=" hurts ")], None


@pytest.fixture
def stt(monkeypatch: pytest.MonkeyPatch) -> adapter.FasterWhisperSTT:
    FakeModel.instances.clear()
    monkeypatch.setattr(adapter, "WhisperModel", FakeModel)
    return adapter.FasterWhisperSTT("small", "cpu", "int8")


def test_the_model_is_loaded_with_the_chosen_device_and_precision(
    stt: adapter.FasterWhisperSTT,
) -> None:
    assert isinstance(stt, SpeechToText)
    assert FakeModel.instances[0].args == ("small", "cpu", "int8")


def test_pcm_becomes_float_audio_and_the_language_is_given_not_guessed(
    stt: adapter.FasterWhisperSTT,
) -> None:
    pcm = np.array([0, 16384, -32768, 32767], dtype=np.int16).tobytes()
    text = asyncio.run(stt.transcribe(pcm, 16_000, "ru"))
    assert text == "my tooth hurts"
    ((audio, options),) = FakeModel.instances[0].calls
    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])
    assert (options["language"], options["vad_filter"], options["condition_on_previous_text"]) == (
        "ru",
        False,
        False,
    )


def test_other_sample_rates_are_refused_not_resampled(stt: adapter.FasterWhisperSTT) -> None:
    with pytest.raises(ValueError, match="16000 Hz"):
        asyncio.run(stt.transcribe(b"\x00\x00", 22_050, "en"))
