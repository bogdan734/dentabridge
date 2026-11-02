"""The adapters against the real services on this machine: Ollama, a Piper HTTP server, whisper.

Slow and skipped when a service is not there. Start Piper with
`python -m piper.http_server -m en_US-ljspeech-medium --data-dir <voices> --port 5100`.
"""

import asyncio

import httpx
import numpy as np
import pytest

from dentabridge_core import GlossaryEntry
from dentabridge_voice.adapters.ollama_translator import OllamaTranslator
from dentabridge_voice.adapters.piper_http import PiperHttpTTS

pytestmark = pytest.mark.slow
OLLAMA = "http://localhost:11434"
PIPER = "http://127.0.0.1:5100"
VOICES = {"en": "en_US-ljspeech-medium", "ru": "ru_RU-denis-medium"}


def _up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code < 500
    except httpx.HTTPError:
        return False


def resample(pcm: bytes, source: int, target: int) -> bytes:
    """Linear interpolation: crude, and enough to hand synthetic speech to a recogniser."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    positions = np.linspace(0, len(samples) - 1, int(len(samples) * target / source))
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.int16).tobytes()


def test_ollama_translates_with_the_glossary() -> None:
    if not _up(f"{OLLAMA}/api/version"):
        pytest.skip("Ollama is not running")
    tags = httpx.get(f"{OLLAMA}/api/tags").json()["models"]
    if not any(m["name"].startswith("qwen3:4b-instruct") for m in tags):
        pytest.skip("ollama pull qwen3:4b-instruct")
    translator = OllamaTranslator(OLLAMA, "qwen3:4b-instruct")
    glossary = [GlossaryEntry(src="tooth|teeth", tgt="зуб|зуба")]
    result = asyncio.run(translator.translate("Tooth 36 hurts.", "en", "ru", glossary))
    assert "36" in result.text
    assert "зуб" in result.text.lower()


def test_piper_speaks_english_and_russian_at_the_expected_rate() -> None:
    if not _up(f"{PIPER}/voices"):
        pytest.skip("no Piper server on :5100")
    engine = PiperHttpTTS(PIPER, VOICES)

    async def speak(text: str, lang: str) -> bytes:
        return b"".join([chunk async for chunk in engine.synthesise(text, lang)])

    for text, lang in (("My tooth hurts.", "en"), ("У меня болит зуб.", "ru")):
        pcm = asyncio.run(speak(text, lang))
        seconds = len(pcm) / 2 / engine.sample_rate
        assert 0.5 < seconds < 6, (lang, seconds)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        assert np.sqrt(np.mean(samples**2)) > 200, "the audio is not silence"


def test_what_piper_says_whisper_hears() -> None:
    pytest.importorskip("faster_whisper")
    if not _up(f"{PIPER}/voices"):
        pytest.skip("no Piper server on :5100")
    from dentabridge_voice.adapters.faster_whisper import FasterWhisperSTT

    engine = PiperHttpTTS(PIPER, VOICES)

    async def speak(text: str, lang: str) -> bytes:
        return b"".join([chunk async for chunk in engine.synthesise(text, lang)])

    stt = FasterWhisperSTT("small", "cpu", "int8")
    for text, lang, expected in (
        ("My tooth hurts when I drink something cold.", "en", "tooth"),
        ("У меня болит зуб, когда я пью холодное.", "ru", "зуб"),
    ):
        pcm22 = asyncio.run(speak(text, lang))
        pcm16 = resample(pcm22, engine.sample_rate, 16_000)
        heard = asyncio.run(stt.transcribe(pcm16, 16_000, lang))
        assert expected in heard.lower(), (lang, heard)
