"""The worker in a real LiveKit room: the server in Docker on 127.0.0.1, the two people as Python
clients on the same SDK the worker uses (dentabridge_voice.testing). The patient says the demo
replay's first patient turn and types one message; the dentist must get the captions, the
translated voice and the translated chat.

Slow, skipped without Docker. "fake-models" checks the room plumbing with scripted models;
"local-models" runs Ollama, Piper (on :5100) and whisper too, and is skipped when they are not up.
"""

import asyncio
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from uuid import UUID

import httpx
import numpy as np
import pytest

pytest.importorskip("livekit.rtc")
pytest.importorskip("soundfile")
pytest.importorskip("faster_whisper")
from livekit import api
from livekit.protocol.room import RoomConfiguration

from dentabridge_core import CallRoomMetadata, GlossaryEntry, Translation
from dentabridge_voice.adapters.piper_http import VOICES, PiperHttpTTS
from dentabridge_voice.rooms import LiveKitServer, watch_rooms
from dentabridge_voice.testing import Dentist, assert_interpreted, call
from dentabridge_voice.worker import Services, UtcClock

pytestmark = pytest.mark.slow
IMAGE = "livekit/livekit-server:v1.13.10"
KEY, SECRET = "dentabridge", "e2e-secret-long-enough-for-hs256-signing"
SESSION = UUID("00000000-0000-4000-8000-0000000000e2")
ROOM = "db-e2e"
META = CallRoomMetadata.model_validate(
    {
        "session_id": SESSION,
        "languages": {"patient": "ru", "dentist": "en"},
        "mode": "consecutive",
        "glossary_id": "dental-en-ru",
    }
)
TYPED = "Болит зуб 36"


class Hearing:
    async def transcribe(self, pcm: bytes, sample_rate: int, lang: str) -> str:
        return TYPED


class Translating:
    model_id = "scripted"

    async def translate(
        self, text: str, src: str, tgt: str, glossary: Sequence[GlossaryEntry]
    ) -> Translation:
        return Translation(text="My tooth 36 hurts.", latency_ms=1.0)


class Tone:
    sample_rate = 22_050

    async def synthesise(self, text: str, lang: str) -> AsyncIterator[bytes]:
        t = np.arange(2205) / self.sample_rate
        for _ in range(5):
            yield (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16).tobytes()


def _up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code < 500
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def server() -> Iterator[LiveKitServer]:
    if shutil.which("docker") is None:
        pytest.skip("needs Docker for the LiveKit server")
    name = "dentabridge-livekit-e2e"
    ports = [
        "-p",
        "127.0.0.1:7880:7880",
        "-p",
        "127.0.0.1:7881:7881",
        "-p",
        "127.0.0.1:7882:7882/udp",
    ]
    flags = ["--keys", f"{KEY}: {SECRET}", "--bind", "0.0.0.0", "--node-ip", "127.0.0.1"]
    command = ["docker", "run", "-d", "--rm", "--name", name, *ports, IMAGE, *flags]
    started = subprocess.run([*command, "--udp-port", "7882"], capture_output=True, text=True)
    if started.returncode != 0:
        pytest.skip(f"LiveKit did not start: {started.stderr.strip()[:300]}")
    try:
        deadline = time.monotonic() + 30
        while not _up("http://127.0.0.1:7880"):
            if time.monotonic() > deadline:
                pytest.fail("LiveKit did not answer within 30 s")
            time.sleep(0.5)
        yield LiveKitServer("ws://127.0.0.1:7880", KEY, SECRET)
    finally:
        subprocess.run(["docker", "stop", name], capture_output=True, check=False)


@pytest.fixture(params=["fake-models", "local-models"])
def services(request: pytest.FixtureRequest) -> Services:
    if request.param == "fake-models":
        return Services(stt=Hearing(), translator=Translating(), tts=Tone(), wall=UtcClock())
    from dentabridge_voice.adapters.faster_whisper import FasterWhisperSTT
    from dentabridge_voice.adapters.ollama_translator import OllamaTranslator

    if not _up("http://localhost:11434/api/version"):
        pytest.skip("Ollama is not running")
    if not _up("http://127.0.0.1:5100/voices"):
        pytest.skip("no Piper server on :5100")
    return Services(
        stt=FasterWhisperSTT("small", "cpu", "int8"),
        translator=OllamaTranslator("http://localhost:11434", "qwen3:4b-instruct"),
        tts=PiperHttpTTS("http://127.0.0.1:5100", VOICES),
        wall=UtcClock(),
    )


def person_token(role: str) -> str:
    """What the API mints: the person's identity, and the call metadata the room opens with."""
    grants = api.VideoGrants(
        room_join=True, room=ROOM, can_publish=True, can_subscribe=True, can_publish_data=True
    )
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(f"{role}-{SESSION}")
        .with_grants(grants)
        .with_room_config(RoomConfiguration(metadata=META.model_dump_json()))
        .to_jwt()
    )


async def interpreted_call(server: LiveKitServer, services: Services) -> Dentist:
    worker = asyncio.create_task(watch_rooms(server, services))
    try:
        patient, dentist = person_token("patient"), person_token("dentist")
        return await call(server.url, patient, dentist, f"interpreter-{SESSION}", TYPED)
    finally:
        worker.cancel()


def test_the_dentist_reads_and_hears_the_patient_in_english(
    server: LiveKitServer, services: Services
) -> None:
    assert_interpreted(asyncio.run(interpreted_call(server, services)), TYPED)
