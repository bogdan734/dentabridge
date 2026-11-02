import asyncio
import io
import json
import wave
from collections.abc import Callable

import httpx
import pytest

from dentabridge_voice.adapters.piper_http import PiperHttpTTS
from dentabridge_voice.ports import TextToSpeech

VOICES = {"en": "en_US-ljspeech-medium", "ru": "ru_RU-denis-medium"}
RATE = 22_050


def wav(seconds: float = 0.25, rate: int = RATE, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x01\x00" * int(seconds * rate) * channels)
    return buffer.getvalue()


def server(
    audio: bytes, available: tuple[str, ...] = tuple(VOICES.values())
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/voices":
            return httpx.Response(200, json={name: {} for name in available})
        return httpx.Response(200, content=audio, headers={"content-type": "audio/wav"})

    return handler, seen


def tts(handler: Callable[[httpx.Request], httpx.Response]) -> PiperHttpTTS:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return PiperHttpTTS("http://piper.test", VOICES, client=client)


def speak(engine: PiperHttpTTS, text: str = "My tooth hurts.", lang: str = "en") -> list[bytes]:
    async def collect() -> list[bytes]:
        return [chunk async for chunk in engine.synthesise(text, lang)]

    return asyncio.run(collect())


def test_piper_mock_transport() -> None:
    handler, seen = server(wav(0.25))
    engine = tts(handler)
    assert isinstance(engine, TextToSpeech)
    speak(engine)
    synth = [r for r in seen if r.url.path == "/synthesize"]
    assert len(synth) == 1
    assert json.loads(synth[0].content) == {
        "text": "My tooth hurts.",
        "voice": "en_US-ljspeech-medium",
    }
    assert engine.sample_rate == RATE


def test_audio_comes_in_100_ms_pieces_and_adds_up() -> None:
    handler, _ = server(wav(0.25))
    chunks = speak(tts(handler))
    pcm_bytes = 2 * int(0.25 * RATE)
    assert all(len(c) == 4410 for c in chunks[:-1])
    assert 0 < len(chunks[-1]) <= 4410
    assert sum(map(len, chunks)) == pcm_bytes


def test_each_language_gets_its_own_voice() -> None:
    handler, seen = server(wav(0.1))
    engine = tts(handler)
    speak(engine, lang="en")
    speak(engine, "У меня болит зуб.", lang="ru")
    voices = [json.loads(r.content)["voice"] for r in seen if r.url.path == "/synthesize"]
    assert voices == ["en_US-ljspeech-medium", "ru_RU-denis-medium"]


def test_the_server_is_asked_which_voices_it_has_only_once() -> None:
    handler, seen = server(wav(0.1))
    engine = tts(handler)
    speak(engine)
    speak(engine)
    assert len([r for r in seen if r.url.path == "/voices"]) == 1


def test_a_voice_the_server_lacks_fails_loudly_instead_of_a_silent_fallback() -> None:
    handler, _ = server(wav(0.1), available=("en_US-ljspeech-medium",))
    with pytest.raises(ValueError, match="ru_RU-denis-medium"):
        speak(tts(handler), "У меня болит зуб.", "ru")


def test_a_language_without_a_voice_is_refused() -> None:
    handler, _ = server(wav(0.1))
    with pytest.raises(ValueError, match="no voice for de"):
        speak(tts(handler), "Guten Tag", "de")


@pytest.mark.parametrize("audio", [wav(0.1, rate=16_000), wav(0.1, channels=2)])
def test_audio_in_another_format_is_refused(audio: bytes) -> None:
    handler, _ = server(audio)
    with pytest.raises(ValueError, match="22050 Hz mono"):
        speak(tts(handler))


def test_a_server_error_is_an_error() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/voices":
            return httpx.Response(200, json={v: {} for v in VOICES.values()})
        return httpx.Response(500, text="boom")

    with pytest.raises(httpx.HTTPStatusError):
        speak(tts(broken))
