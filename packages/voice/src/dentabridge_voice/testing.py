"""Two people for the call tests: a patient who says the demo replay's first patient turn and types
one message, and a dentist who collects what reaches them. Needs the `worker` and `harness`
extras, a LiveKit server and an interpreter serving it.
"""

import asyncio
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import numpy as np
import soundfile  # type: ignore[import-untyped]  # ships no type information
from livekit import rtc
from numpy.typing import NDArray

from dentabridge_core import CaptionEvent, ChatEvent, ReplayManifest
from dentabridge_voice.worker import CAPTIONS, CHAT_IN, CHAT_OUT

DEMO = Path(__file__).resolve().parents[2] / "replays" / "demo"


def patient_turn() -> tuple[NDArray[np.int16], int]:
    """The end of the demo replay's first patient turn (synthetic Russian speech) and the 1.5 s
    of silence after it, at 16 kHz."""
    manifest = ReplayManifest.model_validate_json((DEMO / "manifest.json").read_text())
    end_ms = next(e for e in manifest.events if e.speaker == "patient").t_speech_end_ms
    samples, rate = soundfile.read(DEMO / "audio" / "patient-ru.ogg", dtype="int16")
    return samples[max(0, end_ms - 5_000) * rate // 1000 : (end_ms + 1_500) * rate // 1000], rate


class Dentist:
    """What reaches the dentist: captions, chat, and how loud the patient's translated voice is."""

    def __init__(self, room: rtc.Room, interpreter: str) -> None:
        self.captions: list[CaptionEvent] = []
        self.chat: list[ChatEvent] = []
        self.loudest = 0
        self.interpreter_joined = asyncio.Event()
        self._interpreter = interpreter
        self._tasks: set[asyncio.Task[None]] = set()
        room.register_text_stream_handler(CAPTIONS, self._on_caption)
        room.register_text_stream_handler(CHAT_OUT, self._on_chat)
        room.on("track_subscribed", self._on_track)
        room.on("participant_connected", self._on_participant)

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_participant(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self._interpreter:
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

    def done(self) -> bool:
        return bool(self.spoken() and self.loudest > 2000 and self.chat)


async def until(done: Callable[[], bool], seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        await asyncio.sleep(0.2)


async def call(
    url: str, patient_token: str, dentist_token: str, interpreter: str, typed: str
) -> Dentist:
    """The dentist joins, the patient joins, waits for the interpreter, speaks and types `typed`;
    returns what reached the dentist within 90 s."""
    dentist_room, patient_room = rtc.Room(), rtc.Room()
    dentist = Dentist(dentist_room, interpreter)
    try:
        await dentist_room.connect(url, dentist_token)
        await patient_room.connect(url, patient_token)
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
        await patient_room.local_participant.send_text(typed, topic=CHAT_IN)
        await until(dentist.done, 90)
        return dentist
    finally:
        await patient_room.disconnect()
        await dentist_room.disconnect()


def assert_interpreted(dentist: Dentist, typed: str) -> None:
    """The patient's Russian reached the dentist as an English caption, translated speech and a
    translated chat message, and the source caption came first."""
    final = dentist.spoken()
    assert final is not None, [c.model_dump() for c in dentist.captions]
    assert (final.speaker, final.src_lang, final.tgt_lang, final.audio) == (
        "patient",
        "ru",
        "en",
        "spoken",
    ), final
    assert final.src_text, final
    assert final.tgt_text, final
    first = next(c for c in dentist.captions if c.utterance_id == final.utterance_id)
    assert (first.final, first.tgt_text) == (False, ""), "the source caption comes first"
    assert dentist.loudest > 2000, "the translated voice is heard, not silence"
    (message,) = dentist.chat
    assert (message.sender, message.src_text, message.translation_status) == (
        "patient",
        typed,
        "ok",
    ), message
