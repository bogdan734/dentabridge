import asyncio
import io
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sacrebleu")
pytest.importorskip("soundfile")
import soundfile  # type: ignore[import-untyped]  # ships no type information
from numpy.typing import NDArray

from dentabridge_core import GlossaryEntry, ReplayManifest, Translation
from dentabridge_voice.checks import digits_preserved, missing_terms
from dentabridge_voice.glossary import GLOSSARY_DIR, load_glossary
from dentabridge_voice.harness.run import (
    Clip,
    HarnessReport,
    HarnessResult,
    load_script,
    record_replay,
    run_harness,
)
from dentabridge_voice.pipeline import Pipeline
from dentabridge_voice.ports import Utterance

GLOSSARY = load_glossary(GLOSSARY_DIR / "en-ru.tsv")
RATE = 16_000


class Ticking:
    """Milliseconds, advancing 10 per reading: the harness reads the clock once per event."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> float:
        self.now += 10
        return float(self.now)


class FixedMs:
    def now_ms(self) -> int:
        return 0


class ScriptedSTT:
    def __init__(self, heard: dict[bytes, str]) -> None:
        self.heard = heard

    async def transcribe(self, pcm: bytes, sample_rate: int, lang: str) -> str:
        return self.heard[pcm]


class ScriptedTranslator:
    model_id = "scripted"

    def __init__(self, said: dict[str, str]) -> None:
        self.said = said

    async def translate(
        self, text: str, src: str, tgt: str, glossary: Sequence[GlossaryEntry]
    ) -> Translation:
        return Translation(text=self.said[text], latency_ms=1.0)


class Voice:
    sample_rate = 22_050

    async def synthesise(self, text: str, lang: str) -> AsyncIterator[bytes]:
        t = np.arange(2205) / self.sample_rate
        for _ in range(3):  # 100 ms each, a tone loud enough to survive Vorbis
            yield bytes((np.sin(2 * np.pi * 500 * t) * 12000).astype(np.int16).tobytes())


def tone(hz: int) -> bytes:
    """100 ms of a sine, loud enough to survive Vorbis; the pitch tells the clips apart."""
    t = np.arange(1600) / RATE
    return bytes((np.sin(2 * np.pi * hz * t) * 12000).astype(np.int16).tobytes())


def clips(n: int = 4) -> list[Clip]:
    out = []
    for i in range(n):
        english = i % 2 == 0
        out.append(
            Clip(
                utterance=Utterance(
                    utterance_id=f"u{i}",
                    speaker="dentist" if english else "patient",
                    pcm=tone(300 + 100 * i),
                    sample_rate=RATE,
                    t_speech_end_ms=0,
                ),
                src="en" if english else "ru",
                tgt="ru" if english else "en",
                reference_text=f"clip {i} tooth {i}0" if english else f"клип {i} зуб {i}0",
                reference_translation=f"клип {i} зуб {i}0" if english else f"clip {i} tooth {i}0",
            )
        )
    return out


def run(
    clip_list: list[Clip], heard: dict[bytes, str] | None = None, said: dict[str, str] | None = None
) -> HarnessResult:
    heard = heard or {c.utterance.pcm: c.reference_text for c in clip_list}
    said = said or {c.reference_text: c.reference_translation for c in clip_list}
    pipeline = Pipeline(ScriptedSTT(heard), ScriptedTranslator(said), Voice(), GLOSSARY, FixedMs())
    return asyncio.run(run_harness(pipeline, clip_list, GLOSSARY, clock=Ticking()))


def test_the_script_is_50_turns_whose_pairs_pass_our_own_checks() -> None:
    script = load_script()
    assert len(script) == 50
    assert {(line.speaker, line.lang) for line in script} == {("patient", "ru"), ("dentist", "en")}
    for line in script:
        src, tgt = ("en", "ru") if line.lang == "en" else ("ru", "en")
        entries = GLOSSARY.for_direction(src, tgt)
        assert missing_terms(line.text, line.translation, entries) == (), line.text
        assert digits_preserved(line.text, line.translation), line.text


def test_a_perfect_run_has_no_errors_and_measured_latency() -> None:
    result = run(clips())
    report = result.report
    assert (report.n, report.n_silent) == (4, 0)
    assert (report.wer, report.chrf, report.glossary_acc, report.digit_acc) == (0, 100, 1, 1)
    # one reading at the start, then one per event: partial, final, first audio -> 10 ms each
    assert report.stage_ms == {"stt": 10, "translate": 10, "tts_first": 10}
    assert (report.p50_ms, report.p95_ms) == (30, 30)


def test_report_schema() -> None:
    report = run(clips()).report
    document = json.loads(report.model_dump_json())
    assert set(document) == {
        "n",
        "n_silent",
        "stage_ms",
        "p50_ms",
        "p95_ms",
        "wer",
        "chrf",
        "glossary_acc",
        "digit_acc",
    }
    assert HarnessReport.model_validate(document) == report


def test_mistakes_show_up_in_the_numbers() -> None:
    clip_list = clips()
    heard = {c.utterance.pcm: c.reference_text for c in clip_list}
    heard[clip_list[0].utterance.pcm] = "clip 0 teeth 00"  # a misheard digit pair and a word
    said = {c.reference_text: c.reference_translation for c in clip_list}
    said["clip 0 teeth 00"] = "клип 0 зуб 99"  # a swapped number
    report = run(clip_list, heard, said).report
    assert report.wer > 0
    assert report.chrf < 100
    assert report.digit_acc is not None
    assert report.digit_acc < 1


def test_a_silent_clip_counts_as_a_miss_and_does_not_hide_the_rest() -> None:
    clip_list = clips()
    heard = {c.utterance.pcm: c.reference_text for c in clip_list}
    heard[clip_list[1].utterance.pcm] = ""
    result = run(clip_list, heard)
    assert (result.report.n, result.report.n_silent) == (4, 1)
    assert result.report.wer > 0
    assert result.report.p50_ms == 30


def test_deterministic_with_fakes() -> None:
    first, second = run(clips()), run(clips())
    assert first.report == second.report
    assert [r.events for r in first.runs] == [r.events for r in second.runs]


def test_replay_matches_core_schema(tmp_path: Path) -> None:
    result = run(clips())
    manifest = record_replay(result.runs, tmp_path)
    assert ReplayManifest.model_validate_json((tmp_path / "manifest.json").read_text()) == manifest
    assert manifest.replay_id == tmp_path.name
    assert {(t.role, t.lang) for t in manifest.tracks} == {
        ("patient", "ru"),
        ("dentist", "en"),
        ("patient", "en"),
        ("dentist", "ru"),
    }
    assert manifest.n == 4
    assert [e.utterance_id for e in manifest.events] == ["u0", "u1", "u2", "u3"]
    assert all(e.final and e.t_first_audio_ms is not None for e in manifest.events)
    ends = [e.t_speech_end_ms for e in manifest.events]
    assert ends == sorted(ends)
    assert all(e.t_first_audio_ms > e.t_speech_end_ms for e in manifest.events)  # type: ignore[operator]
    assert manifest.latency_p50_ms == manifest.latency_p95_ms == 30


def test_replay_audio_files_are_ogg_tracks_of_one_length_with_speech_where_it_belongs(
    tmp_path: Path,
) -> None:
    manifest = record_replay(run(clips()).runs, tmp_path)

    def read(role: str, lang: str) -> NDArray[np.float64]:
        track = next(t for t in manifest.tracks if (t.role, t.lang) == (role, lang))
        data, rate = soundfile.read(io.BytesIO((tmp_path / track.url).read_bytes()))
        assert rate == RATE
        return np.asarray(data, dtype=np.float64)

    tracks = {
        key: read(*key)
        for key in (("patient", "ru"), ("dentist", "en"), ("patient", "en"), ("dentist", "ru"))
    }
    assert len({len(data) for data in tracks.values()}) == 1, "every track covers the same timeline"

    def window(event_index: int) -> slice:  # the 100 ms the speaker of that turn talked
        end = int(manifest.events[event_index].t_speech_end_ms / 1000 * RATE)
        return slice(end - 1600, end)

    assert np.abs(tracks[("dentist", "en")][window(0)]).max() > 0.01, "the dentist speaks first"
    assert np.abs(tracks[("dentist", "en")][window(1)]).max() < 0.01, (
        "and is quiet in the patient's turn"
    )
    assert np.abs(tracks[("patient", "ru")][window(1)]).max() > 0.01
    assert np.abs(tracks[("patient", "ru")][window(0)]).max() < 0.01
    first_audio = int(manifest.events[0].t_first_audio_ms / 1000 * RATE)  # type: ignore[operator]
    assert np.abs(tracks[("dentist", "ru")][first_audio : first_audio + 1600]).max() > 0.01
