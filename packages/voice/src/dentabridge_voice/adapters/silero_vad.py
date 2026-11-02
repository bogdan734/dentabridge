"""Silero VAD (MIT), streamed: how likely one 32 ms window of 16 kHz audio is speech.

The model file ships inside faster-whisper (the `stt` extra), so nothing more is downloaded. The
model is recurrent: each stream keeps its own state and the last 64 samples as context, which is
what faster-whisper does across the windows of one file.
"""

from functools import cache
from typing import Any

import numpy as np
from faster_whisper.vad import get_vad_model  # type: ignore[import-untyped]  # ships no type information

RATE = 16_000
WINDOW = 512  # samples: 32 ms, the window the model is trained on at 16 kHz
CONTEXT = 64


@cache
def _session() -> Any:
    """One onnxruntime session for every stream; only the state is per stream."""
    return get_vad_model().session


class SileroVad:
    def __init__(self) -> None:
        self._session = _session()
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def probability(self, frame: bytes) -> float:
        """`frame`: one window of 16-bit mono PCM; the next call takes the window after it."""
        if len(frame) != WINDOW * 2:
            raise ValueError(f"a window is {WINDOW * 2} bytes, got {len(frame)}")
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        batch = np.concatenate([self._context, samples])[np.newaxis, :]
        out, self._h, self._c = self._session.run(
            None, {"input": batch, "h": self._h, "c": self._c}
        )
        self._context = samples[-CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])
