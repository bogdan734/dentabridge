"""The worker in a real LiveKit room: the server in Docker on 127.0.0.1, the two people as Python
clients on the same SDK the worker uses. The patient says the first patient turn of the demo
replay and types one chat message; the dentist must get the captions, the translated voice and
the translated chat.

Slow, skipped without Docker. "fake-models" checks the room plumbing with scripted models;
"local-models" runs Ollama, Piper (on :5100) and whisper too, and is skipped when they are not up.
"""

import asyncio
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import numpy as np
import pytest

pytest.importorskip("livekit.rtc")
pytest.importorskip("soundfile")
pytest.importorskip("faster_whisper")
import soundfile  # type: ignore[import-untyped]  # ships no type information
from livekit import api, rtc
from livekit.protocol.room import RoomConfiguration
from numpy.typing import NDArray

from dentabridge_core import (
    CallRoomMetadata,
    CaptionEvent,
    ChatEvent,
    GlossaryEntry,
    ReplayManifest,
    Translation,
)
from dentabridge_voice.adapters.piper_http import VOICES, PiperHttpTTS
from dentabridge_voice.rooms import LiveKitServer, watch_rooms
from dentabridge_voice.worker import CAPTIONS, CHAT_IN, CHAT_OUT, Services

pytestmark = pytest.mark.slow
IMAGE = "livekit/livekit-server:v1.13.10"
KEY, SECRET = "dentabridge", "e2e-secret-long-enough-for-hs256-signing"
SESSION = UUID("00000000-0000-4000-8000-0000000000e2")
ROOM = "db-e2e"
DEMO = Path(__file__).resolve().parents[1] / "replays" / "demo"
META = CallRoomMetadata.model_validate(
    {
        "session_id": SESSION,
        "languages": {"patient": "ru", "dentist": "en"},
        "mode": "consecutive",
        "glossary_id": "dental-en-ru",
    }
)
TYPED = "Болит зуб 36"


class UtcClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


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


def patient_turn() -> tuple[NDArray[np.int16], int]:
    """The end of the demo replay's first patient turn (synthetic Russian speech) and the 1.5 s
    of silence after it, 16 kHz."""
    manifest = ReplayManifest.model_validate_json((DEMO / "manifest.json").read_text())
    end_ms = next(e for e in manifest.events if e.speaker == "patient").t_speech_end_ms
    samples, rate = soundfile.read(DEMO / "audio" / "patient-ru.ogg", dtype="int16")
    return samples[max(0, end_ms - 5_000) * rate // 1000 : (end_ms + 1_500) * rate // 1000], rate


class Dentist:
    """Collects what reaches the dentist: captions, chat, and how loud the translated voice is."""

    def __init__(self, room: rtc.Room) -> None:
        self.captions: list[CaptionEvent] = []
        self.chat: list[ChatEvent] = []
        self.loudest = 0
        self.interpreter_joined = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        room.register_text_stream_handler(CAPTIONS, self._on_caption)
        room.register_text_stream_handler(CHAT_OUT, self._on_chat)
        room.on("track_subscribed", self._on_track)
        room.on("participant_connected", self._on_participant)

    def _spawn(self, work: Any) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_participant(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == f"interpreter-{SESSION}":
            self.interpreter_joined.set()

    def _on_caption(self, reader: rtc.TextStreamReader, identity: str) -> None:
        async def read() -> None:
            self.captions.append(CaptionEvent.model_validate_json(await reader.read_all()))

        self._spawn(read())

    def _on_chat(self, reader: rtc.TextStreamReader, identity: str) -> None:
        async def read() -> None:
            self.chat.append(ChatEvent.model_validate_json(await reader.read_all()))

        self._spawn(read())

    def _on_track(
        self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: Any
    ) -> None:
        if publication.name != "translated-patient":
            return

        async def listen() -> None:
            async for event in rtc.AudioStream.from_track(track=track, sample_rate=16_000):
                samples = np.frombuffer(event.frame.data, dtype=np.int16)
                self.loudest = max(self.loudest, int(np.abs(samples).max(initial=0)))

        self._spawn(listen())

    def spoken(self) -> CaptionEvent | None:
        return next((c for c in self.captions if c.final and c.t_first_audio_ms), None)


async def until(done: Callable[[], bool], seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not done():
        if time.monotonic() > deadline:
            return
        await asyncio.sleep(0.2)


async def call(server: LiveKitServer, services: Services) -> Dentist:
    worker = asyncio.create_task(watch_rooms(server, services))
    dentist_room, patient_room = rtc.Room(), rtc.Room()
    dentist = Dentist(dentist_room)
    try:
        await dentist_room.connect(server.url, person_token("dentist"))
        await patient_room.connect(server.url, person_token("patient"))
        microphone = rtc.AudioSource(16_000, 1)
        track = rtc.LocalAudioTrack.create_audio_track("microphone", microphone)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await patient_room.local_participant.publish_track(track, options)
        await asyncio.wait_for(dentist.interpreter_joined.wait(), 15)
        await asyncio.sleep(1.5)  # the interpreter subscribes to the patient's microphone
        samples, rate = patient_turn()
        step = rate // 100  # 10 ms frames, paced by the source's queue like a live microphone
        for i in range(0, len(samples) - step + 1, step):
            frame = rtc.AudioFrame(samples[i : i + step].tobytes(), rate, 1, step)
            await microphone.capture_frame(frame)
        await patient_room.local_participant.send_text(TYPED, topic=CHAT_IN)
        await until(lambda: bool(dentist.spoken() and dentist.loudest > 2000 and dentist.chat), 90)
        return dentist
    finally:
        worker.cancel()
        await patient_room.disconnect()
        await dentist_room.disconnect()


def test_the_dentist_reads_and_hears_the_patient_in_english(
    server: LiveKitServer, services: Services
) -> None:
    dentist = asyncio.run(call(server, services))
    final = dentist.spoken()
    assert final is not None, [c.model_dump() for c in dentist.captions]
    assert (final.speaker, final.src_lang, final.tgt_lang, final.audio) == (
        "patient",
        "ru",
        "en",
        "spoken",
    )
    assert final.src_text
    assert final.tgt_text
    first = next(c for c in dentist.captions if c.utterance_id == final.utterance_id)
    assert (first.final, first.tgt_text) == (False, ""), "the source caption comes first"
    assert dentist.loudest > 2000, "the translated voice is heard, not silence"
    (message,) = dentist.chat
    assert (message.sender, message.src_text, message.translation_status) == (
        "patient",
        TYPED,
        "ok",
    )
