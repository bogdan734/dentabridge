"""Audio helpers for the harness: 16 kHz mono floats in, Ogg Vorbis out."""

from pathlib import Path

import numpy as np
import soundfile  # type: ignore[import-untyped]  # ships no type information
from numpy.typing import NDArray

RATE = 16_000


def to_16k(pcm: bytes, rate: int) -> NDArray[np.float32]:
    """16-bit mono PCM at `rate` as floats at 16 kHz (linear interpolation: speech, not music)."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if rate == RATE or len(samples) == 0:
        return samples
    positions = np.linspace(0, len(samples) - 1, round(len(samples) * RATE / rate))
    return np.asarray(np.interp(positions, np.arange(len(samples)), samples), dtype=np.float32)


def to_pcm16(samples: NDArray[np.float32]) -> bytes:
    return bytes((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


def write_ogg(path: Path, samples: NDArray[np.float32]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, samples, RATE, format="OGG", subtype="VORBIS")
