"""The interpreter inside one call room, on the LiveKit rtc SDK (the `worker` extra).

It subscribes to the microphones of the call's two people, never to an agent's track or a
translated one, and runs each through the speaker loop. What it hears goes out as captions on the
text stream `captions.v1` and as speech on the audio track `translated-<role>`: what <role> said,
in the other person's language. Chat comes in on `chat.in.v1` and goes out translated on
`chat.v1`. It leaves when nobody of the call has been in the room for EMPTY_GRACE_S.
"""

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from livekit import rtc

from dentabridge_core import CallRoomMetadata, CaptionEvent, Role, Translator
from dentabridge_voice.adapters.silero_vad import SileroVad
from dentabridge_voice.chat import ChatTranslator, relay_chat
from dentabridge_voice.glossary import glossary_for
from dentabridge_voice.pipeline import Pipeline
from dentabridge_voice.ports import AudioChunk, SpeechToText, TextToSpeech, WallClock
from dentabridge_voice.routing import (
    Direction,
    ParticipantInfo,
    direction_of,
    route,
    should_subscribe,
    spoken,
    translated_track,
)
from dentabridge_voice.speaker import RATE, run_speaker

CAPTIONS = "captions.v1"
CHAT_IN = "chat.in.v1"
CHAT_OUT = "chat.v1"
EMPTY_GRACE_S = 30.0  # someone whose connection drops and comes back keeps their interpreter

log = logging.getLogger("dentabridge_voice")


@dataclass(frozen=True)
class Services:
    """The models, shared by every room this worker serves."""

    stt: SpeechToText
    translator: Translator
    tts: TextToSpeech | None  # None: every call gets captions only
    wall: WallClock


class UtcClock:
    """Calendar time, for the moment a chat message was sent."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class CallClock:
    """Milliseconds since the interpreter joined: the timeline its captions are stamped on."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def now_ms(self) -> int:
        return round((time.monotonic() - self._start) * 1000)


class RoomOutlet:
    def __init__(self, room: rtc.Room, voice: rtc.AudioSource | None) -> None:
        self._room = room
        self._voice = voice

    async def caption(self, event: CaptionEvent) -> None:
        await self._room.local_participant.send_text(event.model_dump_json(), topic=CAPTIONS)

    async def audio(self, chunk: AudioChunk) -> None:
        if self._voice is None:
            raise RuntimeError("translated audio for a direction that has no voice")
        frame = rtc.AudioFrame(chunk.pcm, chunk.sample_rate, 1, len(chunk.pcm) // 2)
        await self._voice.capture_frame(frame)


def participant_info(participant: rtc.RemoteParticipant) -> ParticipantInfo:
    is_agent = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    return ParticipantInfo(identity=participant.identity, is_agent=is_agent)


def _report(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        log.error("%s failed", task.get_name(), exc_info=error)


async def serve_room(room: rtc.Room, meta: CallRoomMetadata, services: Services) -> None:
    """Interpret in a connected room until it closes or the call's people have left it."""
    glossary = glossary_for(meta.glossary_id)
    clock = CallClock()
    ids = itertools.count(1)  # utterance ids, unique in the room across reconnects
    chat = ChatTranslator(services.translator, glossary, services.wall)
    voices: dict[Role, rtc.AudioSource] = {}
    speakers: dict[str, asyncio.Task[None]] = {}
    chats: set[asyncio.Task[None]] = set()
    closed = asyncio.Event()
    loop = asyncio.get_running_loop()

    def people() -> list[Direction]:
        return route([participant_info(p) for p in room.remote_participants.values()], meta)

    async def voice(direction: Direction, tts: TextToSpeech) -> rtc.AudioSource:
        if direction.role not in voices:
            source = rtc.AudioSource(tts.sample_rate, 1)
            track = rtc.LocalAudioTrack.create_audio_track(translated_track(direction.role), source)
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_UNKNOWN)
            await room.local_participant.publish_track(track, options)
            voices[direction.role] = source
        return voices[direction.role]

    async def listen(track: rtc.Track, direction: Direction) -> None:
        tts = services.tts if spoken(direction, meta) else None
        outlet = RoomOutlet(room, await voice(direction, tts) if tts else None)
        pipeline = Pipeline(services.stt, services.translator, tts, glossary, clock)
        stream = rtc.AudioStream.from_track(track=track, sample_rate=RATE, num_channels=1)
        chunks = (event.frame.data.tobytes() async for event in stream)
        try:
            await run_speaker(chunks, SileroVad(), pipeline, direction, ids, clock, outlet)
        finally:
            await stream.aclose()

    def stop(identity: str) -> None:
        if (task := speakers.pop(identity, None)) is not None:
            task.cancel()

    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        microphone = publication.source == rtc.TrackSource.SOURCE_MICROPHONE
        info = participant_info(participant)
        if should_subscribe(info, publication.name, is_microphone=microphone, meta=meta):
            publication.set_subscribed(True)

    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        direction = direction_of(participant_info(participant), meta)
        if direction is None or track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        stop(participant.identity)
        task = asyncio.create_task(listen(track, direction), name=f"listening to {direction.role}")
        task.add_done_callback(_report)
        speakers[participant.identity] = task

    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        stop(participant.identity)

    def leave_if_nobody() -> None:
        if not people():
            log.info("%s: nobody of the call is left, leaving", room.name)
            closed.set()

    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        stop(participant.identity)
        if not people():
            loop.call_later(EMPTY_GRACE_S, leave_if_nobody)

    async def answer(reader: rtc.TextStreamReader, identity: str) -> None:
        raw = await reader.read_all()
        sender = room.remote_participants.get(identity)
        if sender is None:  # the interpreter's own stream, or someone who already left
            return
        event = await relay_chat(raw, participant_info(sender), meta, chat)
        if event is not None:
            await room.local_participant.send_text(event.model_dump_json(), topic=CHAT_OUT)

    def on_chat(reader: rtc.TextStreamReader, identity: str) -> None:
        task = asyncio.create_task(answer(reader, identity), name="chat message")
        task.add_done_callback(_report)
        chats.add(task)
        task.add_done_callback(chats.discard)

    room.on("track_published", on_track_published)
    room.on("track_subscribed", on_track_subscribed)
    room.on("track_unsubscribed", on_track_unsubscribed)
    room.on("participant_disconnected", on_participant_disconnected)
    room.on("disconnected", lambda reason: closed.set())
    room.register_text_stream_handler(CHAT_IN, on_chat)
    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            on_track_published(publication, participant)
    if not people():
        loop.call_later(EMPTY_GRACE_S, leave_if_nobody)
    try:
        await closed.wait()
    finally:
        for identity in list(speakers):
            stop(identity)
        for task in list(chats):
            task.cancel()
