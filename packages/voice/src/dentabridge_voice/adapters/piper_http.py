"""Text-to-speech through a Piper HTTP server.

Piper is GPL-3.0, so it runs as its own process (`python -m piper.http_server`) and this module only
speaks HTTP to it; nothing of Piper is imported. The server answers a whole WAV, which is handed on
in 100 ms pieces. Piper falls back to its default voice when asked for one it does not have, which
for a Russian sentence would be English phonemes read out as noise, so the voices are checked once.
"""

import io
import wave
from collections.abc import AsyncIterator, Mapping

import httpx

SAMPLE_RATE = 22_050  # the "medium" Piper voices
CHUNK_MS = 100
TIMEOUT_S = 30.0
# the default voices: public-domain and CC0 training data (LICENCES.toml)
VOICES = {"en": "en_US-ljspeech-medium", "ru": "ru_RU-denis-medium"}


class PiperHttpTTS:
    sample_rate = SAMPLE_RATE

    def __init__(
        self, base_url: str, voices: Mapping[str, str], client: httpx.AsyncClient | None = None
    ) -> None:
        self._base = base_url.rstrip("/")
        self._voices = dict(voices)
        self._client = client or httpx.AsyncClient(timeout=TIMEOUT_S)
        self._served: set[str] | None = None

    async def _check(self, voice: str) -> None:
        if self._served is None:
            response = await self._client.get(f"{self._base}/voices")
            response.raise_for_status()
            self._served = set(response.json())
        if voice not in self._served:
            raise ValueError(
                f"the Piper server has no voice {voice}; download it (piper.download_voices)"
            )

    async def synthesise(self, text: str, lang: str) -> AsyncIterator[bytes]:
        voice = self._voices.get(lang)
        if voice is None:
            raise ValueError(f"no voice for {lang}")
        await self._check(voice)
        response = await self._client.post(
            f"{self._base}/synthesize", json={"text": text, "voice": voice}
        )
        response.raise_for_status()
        with wave.open(io.BytesIO(response.content)) as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
                SAMPLE_RATE,
                1,
                2,
            ):
                raise ValueError(f"{voice} must answer 16-bit {SAMPLE_RATE} Hz mono audio")
            pcm = audio.readframes(audio.getnframes())
        step = SAMPLE_RATE * 2 * CHUNK_MS // 1000
        for start in range(0, len(pcm), step):
            yield pcm[start : start + step]
