"""Speech-to-text with faster-whisper: Whisper models (MIT) on CTranslate2, CPU or CUDA.

Needs the `stt` extra. The model is blocking, so it runs off the event loop. It takes 16 kHz audio
(the worker's frames already are); resampling here would hide a configuration error.
"""

import asyncio
from typing import Literal

import numpy as np
from faster_whisper import WhisperModel  # type: ignore[import-untyped]  # ships no type information

SAMPLE_RATE = 16_000


class FasterWhisperSTT:
    def __init__(self, model: str, device: Literal["cpu", "cuda"], compute_type: str) -> None:
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    async def transcribe(self, pcm: bytes, sample_rate: int, lang: str) -> str:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"faster-whisper takes {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return await asyncio.to_thread(self._run, audio, lang)

    def _run(self, audio: np.ndarray, lang: str) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=lang,  # the speaker's language is known from the call, never guessed
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,  # the segmenter has cut the speech already
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
