"""One person's microphone in; captions and translated speech out, one utterance at a time.

The microphone is cut into utterances while the one before is still being translated (a queue
sits between the two), so a slow translation never makes the interpreter miss what is said next.
A model that fails loses one utterance, never the call: a recogniser failure before anything was
shown is only logged, a translator failure closes the caption the listener already sees as failed.
"""

import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from typing import Protocol

from dentabridge_core import CaptionEvent, Role
from dentabridge_voice.pipeline import Pipeline
from dentabridge_voice.ports import AudioChunk, Clock, Utterance
from dentabridge_voice.routing import Direction
from dentabridge_voice.segment import Segment, Segmenter

RATE = 16_000
FRAME_MS = 32  # one Silero window at 16 kHz
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2
SPEECH = 0.5  # the probability above which a window counts as speech

log = logging.getLogger("dentabridge_voice")


class SpeechDetector(Protocol):
    def probability(self, frame: bytes) -> float: ...


class Outlet(Protocol):
    """Where the translation goes: the room's caption stream and the translated audio track."""

    async def caption(self, event: CaptionEvent) -> None: ...

    async def audio(self, chunk: AudioChunk) -> None: ...


async def windows(chunks: AsyncIterable[bytes], size: int = FRAME_BYTES) -> AsyncIterator[bytes]:
    """Re-cut PCM chunks of any size into frames of exactly `size` bytes; a piece shorter than a
    frame at the very end is dropped."""
    pending = bytearray()
    async for chunk in chunks:
        pending += chunk
        while len(pending) >= size:
            yield bytes(pending[:size])
            del pending[:size]


async def utterances(
    frames: AsyncIterable[bytes],
    vad: SpeechDetector,
    role: Role,
    ids: Iterator[int],
    clock: Clock,
) -> AsyncIterator[Utterance]:
    """Frames of 16 kHz speech, cut at pauses. An utterance is stamped with the moment its speech
    ended on the call's clock, not the later moment the pause after it was noticed."""
    segmenter = Segmenter(RATE, frame_ms=FRAME_MS)
    heard_ms = 0

    def utterance(segment: Segment) -> Utterance:
        return Utterance(
            utterance_id=f"{role}-{next(ids)}",
            speaker=role,
            pcm=segment.pcm,
            sample_rate=RATE,
            t_speech_end_ms=max(0, clock.now_ms() - (heard_ms - segment.end_ms)),
        )

    async for frame in frames:
        heard_ms += FRAME_MS
        for segment in segmenter.push(frame, vad.probability(frame) > SPEECH):
            yield utterance(segment)
    for segment in segmenter.flush():
        yield utterance(segment)


async def interpret(
    utt: Utterance, pipeline: Pipeline, direction: Direction, outlet: Outlet
) -> None:
    shown: CaptionEvent | None = None
    try:
        async for item in pipeline.handle(utt, direction.src, direction.tgt):
            if isinstance(item, CaptionEvent):
                shown = item
                await outlet.caption(item)
            else:
                await outlet.audio(item)
    except Exception as error:  # a model service failed: this utterance is lost, the call goes on
        log.error("utterance %s not translated: %s", utt.utterance_id, type(error).__name__)
        if shown is not None and not shown.final:
            failed = {"final": True, "translation_status": "failed", "audio": "captions_only"}
            await outlet.caption(CaptionEvent.model_validate({**shown.model_dump(), **failed}))


async def run_speaker(
    chunks: AsyncIterable[bytes],
    vad: SpeechDetector,
    pipeline: Pipeline,
    direction: Direction,
    ids: Iterator[int],
    clock: Clock,
    outlet: Outlet,
) -> None:
    """Translate one microphone until it closes, then finish what is still queued."""
    queue: asyncio.Queue[Utterance | None] = asyncio.Queue()

    async def cut() -> None:
        try:
            frames = windows(chunks)
            async for utt in utterances(frames, vad, direction.role, ids, clock):
                queue.put_nowait(utt)
        finally:
            queue.put_nowait(None)

    async def speak() -> None:
        while (utt := await queue.get()) is not None:
            await interpret(utt, pipeline, direction, outlet)

    await asyncio.gather(cut(), speak())
