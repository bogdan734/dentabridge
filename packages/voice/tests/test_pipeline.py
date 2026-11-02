import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from dentabridge_core import CaptionEvent, GlossaryEntry, Translation
from dentabridge_voice.glossary import Glossary
from dentabridge_voice.pipeline import Pipeline
from dentabridge_voice.ports import AudioChunk, Utterance

TOOTH = GlossaryEntry(src="tooth|teeth", tgt="зуб*")
PAIN = GlossaryEntry(src="pain|ache", tgt="бол*")
GLOSSARY = Glossary(langs=("en", "ru"), entries=(TOOTH, PAIN))
UTTERANCE = Utterance(
    utterance_id="u1",
    speaker="patient",
    pcm=b"\x00\x00" * 160,
    sample_rate=16_000,
    t_speech_end_ms=5_000,
)


class StepClock:
    """Every reading is 40 ms later than the one before."""

    def __init__(self, start_ms: int = 5_000) -> None:
        self._now = start_ms

    def now_ms(self) -> int:
        self._now += 40
        return self._now


class FakeSTT:
    def __init__(self, text: str) -> None:
        self.text = text
        self.heard: list[tuple[int, int, str]] = []

    async def transcribe(self, pcm: bytes, sample_rate: int, lang: str) -> str:
        self.heard.append((len(pcm), sample_rate, lang))
        return self.text


class FakeTranslator:
    model_id = "fake"

    def __init__(self, text: str) -> None:
        self.text = text
        self.glossaries: list[Sequence[GlossaryEntry]] = []

    async def translate(
        self, text: str, src: str, tgt: str, glossary: Sequence[GlossaryEntry]
    ) -> Translation:
        self.glossaries.append(glossary)
        return Translation(text=self.text, latency_ms=1.0)


class FakeTTS:
    sample_rate = 22_050

    def __init__(self, chunks: int = 3, fail_after: int | None = None) -> None:
        self.chunks, self.fail_after = chunks, fail_after

    async def synthesise(self, text: str, lang: str) -> AsyncIterator[bytes]:
        for i in range(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("voice engine went away")
            yield b"\x01\x00" * 100


VOICE = FakeTTS()


def run(
    stt_text: str = "my tooth hurts",
    translation: str = "мой зуб болит",
    tts: FakeTTS | None = VOICE,
    src: str = "en",
    tgt: str = "ru",
) -> tuple[list[CaptionEvent | AudioChunk], FakeTranslator]:
    translator = FakeTranslator(translation)
    pipeline = Pipeline(FakeSTT(stt_text), translator, tts, GLOSSARY, StepClock())

    async def collect() -> list[CaptionEvent | AudioChunk]:
        return [event async for event in pipeline.handle(UTTERANCE, src, tgt)]

    return asyncio.run(collect()), translator


def captions(events: Sequence[Any]) -> list[CaptionEvent]:
    return [e for e in events if isinstance(e, CaptionEvent)]


def test_the_source_text_is_shown_before_the_translation_exists() -> None:
    events, _ = run()
    first = events[0]
    assert isinstance(first, CaptionEvent)
    assert (first.final, first.src_text, first.tgt_text) == (False, "my tooth hurts", "")


def test_caption_before_audio() -> None:
    events, _ = run()
    first_audio = next(i for i, e in enumerate(events) if isinstance(e, AudioChunk))
    finals = [e for e in captions(events[:first_audio]) if e.final]
    assert finals
    assert finals[0].tgt_text == "мой зуб болит"
    assert all(e.utterance_id == "u1" for e in events)


def test_a_spoken_translation_is_reported_with_its_first_audio_time() -> None:
    events, _ = run()
    last = captions(events)[-1]
    chunks = [e for e in events if isinstance(e, AudioChunk)]
    assert len(chunks) == 3
    assert (last.final, last.audio, last.t_first_audio_ms) == (True, "spoken", chunks[0].t_ms)
    assert chunks[0].sample_rate == 22_050


def test_tts_failure_emits_captions_only() -> None:
    events, _ = run(tts=FakeTTS(fail_after=0))
    assert not [e for e in events if isinstance(e, AudioChunk)]
    last = captions(events)[-1]
    assert (last.final, last.audio, last.tgt_text) == (True, "captions_only", "мой зуб болит")


def test_a_voice_that_dies_midway_ends_in_captions_only_too() -> None:
    events, _ = run(tts=FakeTTS(chunks=5, fail_after=2))
    assert len([e for e in events if isinstance(e, AudioChunk)]) == 2
    assert captions(events)[-1].audio == "captions_only"


def test_without_a_voice_every_caption_is_captions_only_from_the_start() -> None:
    events, _ = run(tts=None)
    assert {e.audio for e in captions(events)} == {"captions_only"}
    assert not [e for e in events if isinstance(e, AudioChunk)]
    assert [e.final for e in captions(events)] == [False, True]


def test_missing_glossary_term_reported() -> None:
    events, _ = run(translation="мне плохо")
    finals = [e for e in captions(events) if e.final]
    assert [e.glossary_missing for e in finals] == [("tooth",), ("tooth",)]
    assert captions(events)[0].glossary_missing == (), "nothing is claimed before the translation"


def test_digit_swap_flagged() -> None:
    events, _ = run(stt_text="tooth 36 hurts", translation="зуб 63 болит")
    assert [e.digits_ok for e in captions(events) if e.final] == [False, False]
    events, _ = run(stt_text="tooth 36 hurts", translation="зуб 36 болит")
    assert all(e.digits_ok for e in captions(events))


def test_the_translator_gets_only_the_terms_the_sentence_uses() -> None:
    _, translator = run(stt_text="the tooth is fine", translation="зуб в порядке")
    assert translator.glossaries == [(TOOTH,)]


def test_the_glossary_is_turned_around_for_the_other_direction() -> None:
    events, translator = run(stt_text="болит зуб", translation="a tooth pain", src="ru", tgt="en")
    assert translator.glossaries == [
        (
            GlossaryEntry(src="зуб*", tgt="tooth|teeth"),
            GlossaryEntry(src="бол*", tgt="pain|ache"),
        )
    ]
    assert [e.glossary_missing for e in captions(events) if e.final] == [(), ()]
    events, _ = run(stt_text="болит зуб", translation="it is fine", src="ru", tgt="en")
    assert captions(events)[-1].glossary_missing == ("зуб", "бол")


def test_timings_monotonic() -> None:
    events, _ = run()
    times = [e.t_ms for e in events if isinstance(e, AudioChunk)]
    assert times == sorted(times)
    last = captions(events)[-1]
    assert UTTERANCE.t_speech_end_ms < times[0]
    assert last.t_speech_end_ms == UTTERANCE.t_speech_end_ms
    assert last.t_first_audio_ms is not None
    assert last.t_first_audio_ms >= last.t_speech_end_ms


def test_silence_produces_nothing() -> None:
    events, translator = run(stt_text="   ")
    assert events == []
    assert translator.glossaries == []


def test_a_translator_error_is_not_swallowed() -> None:
    class Broken(FakeTranslator):
        async def translate(self, *args: Any) -> Translation:
            raise RuntimeError("model unavailable")

    pipeline = Pipeline(FakeSTT("hello"), Broken("x"), FakeTTS(), GLOSSARY, StepClock())

    async def collect() -> list[Any]:
        return [e async for e in pipeline.handle(UTTERANCE, "en", "ru")]

    with pytest.raises(RuntimeError, match="model unavailable"):
        asyncio.run(collect())
