import asyncio
import itertools
from collections.abc import AsyncIterator, Sequence

from dentabridge_core import CaptionEvent, GlossaryEntry, Translation
from dentabridge_voice.glossary import Glossary
from dentabridge_voice.pipeline import Pipeline
from dentabridge_voice.ports import AudioChunk
from dentabridge_voice.routing import Direction
from dentabridge_voice.speaker import FRAME_BYTES, FRAME_MS, run_speaker, utterances, windows

GLOSSARY = Glossary(langs=("en", "ru"), entries=(GlossaryEntry(src="tooth|teeth", tgt="зуб*"),))
SPEECH, SILENCE = b"\x01\x00" * (FRAME_BYTES // 2), bytes(FRAME_BYTES)
PATIENT = Direction(speaker_identity="patient-x", role="patient", src="ru", tgt="en", verified=True)
TWO_SENTENCES = [(SILENCE, 10), (SPEECH, 40), (SILENCE, 40), (SPEECH, 20), (SILENCE, 40)]


class FrameClock:
    """The call's clock, moved on by the microphone: every frame is 32 ms of real time."""

    def __init__(self, start_ms: int = 5_000) -> None:
        self.now = start_ms

    def now_ms(self) -> int:
        return self.now


class LoudnessVad:
    def probability(self, frame: bytes) -> float:
        return 1.0 if any(frame) else 0.0


async def microphone(
    clock: FrameClock, pattern: Sequence[tuple[bytes, int]]
) -> AsyncIterator[bytes]:
    for frame, count in pattern:
        for _ in range(count):
            clock.now += FRAME_MS
            yield frame
            await asyncio.sleep(0)  # the translation runs while the person talks


class STT:
    """Hears the utterances in order; None is a recogniser that fails."""

    def __init__(self, *texts: str | None) -> None:
        self.texts = list(texts)

    async def transcribe(self, pcm: bytes, sample_rate: int, lang: str) -> str:
        text = self.texts.pop(0)
        if text is None:
            raise ConnectionError("recogniser down")
        return text


class Translator:
    model_id = "fake"

    async def translate(
        self, text: str, src: str, tgt: str, glossary: Sequence[GlossaryEntry]
    ) -> Translation:
        if "fail" in text:
            raise ConnectionError("translator down")
        return Translation(text=f"<{text}>", latency_ms=1.0)


class Voice:
    sample_rate = 22_050

    async def synthesise(self, text: str, lang: str) -> AsyncIterator[bytes]:
        for _ in range(2):
            yield b"\x02\x00" * 2205


class Room:
    def __init__(self) -> None:
        self.sent: list[CaptionEvent | AudioChunk] = []

    async def caption(self, event: CaptionEvent) -> None:
        self.sent.append(event)

    async def audio(self, chunk: AudioChunk) -> None:
        self.sent.append(chunk)


def call(stt: STT, pattern: Sequence[tuple[bytes, int]]) -> list[CaptionEvent | AudioChunk]:
    clock, room = FrameClock(), Room()
    pipeline = Pipeline(stt, Translator(), Voice(), GLOSSARY, clock)
    mic = microphone(clock, pattern)
    asyncio.run(run_speaker(mic, LoudnessVad(), pipeline, PATIENT, itertools.count(1), clock, room))
    return room.sent


def describe(item: CaptionEvent | AudioChunk) -> str:
    if isinstance(item, AudioChunk):
        return f"{item.utterance_id} audio"
    if not item.final:
        return f"{item.utterance_id} source"
    return f"{item.utterance_id} {'spoken' if item.t_first_audio_ms else 'translated'}"


def test_any_chunk_sizes_come_out_as_whole_frames() -> None:
    data = bytes(range(256)) * 22  # 5632 bytes: five frames and a piece too short to hear

    async def chunks() -> AsyncIterator[bytes]:
        start = 0
        for size in (100, 2000, 3, 1945, 1584):
            yield data[start : start + size]
            start += size

    async def cut() -> list[bytes]:
        return [frame async for frame in windows(chunks())]

    frames = asyncio.run(cut())
    assert [len(f) for f in frames] == [FRAME_BYTES] * 5
    assert b"".join(frames) == data[: 5 * FRAME_BYTES]


def test_an_utterance_ends_when_the_speech_did_on_the_calls_clock() -> None:
    clock = FrameClock(start_ms=5_000)

    async def cut() -> list[str]:
        mic = microphone(clock, [(SILENCE, 10), (SPEECH, 40), (SILENCE, 40)])
        found = [
            u async for u in utterances(mic, LoudnessVad(), "patient", itertools.count(7), clock)
        ]
        return [f"{u.utterance_id} {u.t_speech_end_ms} {len(u.pcm)} {u.sample_rate}" for u in found]

    # the pause is noticed 18 frames after the speech stopped; the end is when it stopped
    assert asyncio.run(cut()) == [f"patient-7 {5_000 + 50 * FRAME_MS} {40 * FRAME_BYTES} 16000"]


def test_captions_come_before_the_voice_and_each_sentence_in_turn() -> None:
    sent = call(STT("болит зуб", "и десна"), TWO_SENTENCES)
    assert [describe(item) for item in sent] == [
        "patient-1 source",
        "patient-1 translated",
        "patient-1 audio",
        "patient-1 audio",
        "patient-1 spoken",
        "patient-2 source",
        "patient-2 translated",
        "patient-2 audio",
        "patient-2 audio",
        "patient-2 spoken",
    ]


def test_a_failed_translation_is_said_and_the_call_goes_on() -> None:
    sent = call(STT("fail, please", "и десна"), TWO_SENTENCES)
    first = [e for e in sent if isinstance(e, CaptionEvent) and e.utterance_id == "patient-1"]
    assert [(e.final, e.translation_status, e.tgt_text, e.audio) for e in first] == [
        (False, "ok", "", "spoken"),
        (True, "failed", "", "captions_only"),
    ]
    assert describe(sent[-1]) == "patient-2 spoken"


def test_a_failed_recogniser_loses_only_its_own_utterance() -> None:
    sent = call(STT(None, "и десна"), TWO_SENTENCES)
    assert {item.utterance_id for item in sent} == {"patient-2"}


def test_what_is_said_as_the_microphone_closes_is_still_translated() -> None:
    sent = call(STT("до свидания"), [(SPEECH, 20)])
    assert describe(sent[-1]) == "patient-1 spoken"
