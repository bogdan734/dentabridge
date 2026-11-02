"""Run a scripted consultation through a Pipeline and measure it; record the run as a replay.

Speech end to first translated audio is the number a caller feels. It is split into the stages the
pipeline passes through (recognition, translation, first audio from the voice), read off the moment
each kind of event arrives. Quality is word error rate of the recognised text, chrF++ of the
translation against the reference, and how often the glossary and digit checks held.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from dentabridge_core import CaptionEvent, ReplayManifest, ReplayTrack, Role
from dentabridge_voice.checks import matched_entries
from dentabridge_voice.glossary import Glossary
from dentabridge_voice.harness.audio import RATE, to_16k, write_ogg
from dentabridge_voice.harness.metrics import chrf, percentile, word_errors
from dentabridge_voice.pipeline import Pipeline
from dentabridge_voice.ports import AudioChunk, Utterance

SCRIPT = Path(__file__).with_name("script_en_ru.yaml")
START_MS = 500
GAP_MS = 600


class ScriptLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    speaker: Role
    lang: str
    text: str
    translation: str


def load_script(path: Path = SCRIPT) -> list[ScriptLine]:
    return [ScriptLine.model_validate(item) for item in yaml.safe_load(path.read_text("utf-8"))]


class Clip(BaseModel):
    model_config = ConfigDict(frozen=True)

    utterance: Utterance
    src: str
    tgt: str
    reference_text: str
    reference_translation: str


class HarnessReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    n_silent: int  # clips the pipeline produced nothing for: counted as misses
    stage_ms: dict[str, float]  # mean per stage, over clips that reached audio
    p50_ms: float | None  # speech end -> first translated audio
    p95_ms: float | None
    wer: float
    chrf: float
    glossary_acc: float | None  # of clips whose sentence used a glossary term
    digit_acc: float | None  # of clips whose sentence held a number


@dataclass(frozen=True)
class ClipRun:
    clip: Clip
    events: tuple[CaptionEvent | AudioChunk, ...]
    stt_ms: float | None
    translate_ms: float | None
    tts_first_ms: float | None

    @property
    def total_ms(self) -> float | None:
        parts = (self.stt_ms, self.translate_ms, self.tts_first_ms)
        return None if None in parts else sum(p for p in parts if p is not None)

    @property
    def final(self) -> CaptionEvent | None:
        captions = [e for e in self.events if isinstance(e, CaptionEvent) and e.final]
        return captions[-1] if captions else None

    @property
    def heard(self) -> str:
        first = next((e for e in self.events if isinstance(e, CaptionEvent)), None)
        return first.src_text if first else ""


@dataclass(frozen=True)
class HarnessResult:
    report: HarnessReport
    runs: tuple[ClipRun, ...]


def _now_ms() -> float:
    return time.perf_counter() * 1000


async def _one(pipeline: Pipeline, clip: Clip, clock: Callable[[], float]) -> ClipRun:
    start = clock()
    events: list[CaptionEvent | AudioChunk] = []
    t_heard = t_translated = t_audio = None
    async for event in pipeline.handle(clip.utterance, clip.src, clip.tgt):
        now = clock()
        events.append(event)
        if isinstance(event, AudioChunk):
            t_audio = now if t_audio is None else t_audio
        elif event.final:
            t_translated = now if t_translated is None else t_translated
        else:
            t_heard = now if t_heard is None else t_heard
    if t_heard is None or t_translated is None or t_audio is None:
        return ClipRun(clip, tuple(events), None, None, None)
    return ClipRun(
        clip, tuple(events), t_heard - start, t_translated - t_heard, t_audio - t_translated
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


async def run_harness(
    pipeline: Pipeline,
    clips: Sequence[Clip],
    glossary: Glossary,
    clock: Callable[[], float] = _now_ms,
) -> HarnessResult:
    """Every clip in turn (a call is one speaker at a time). `clock` reads milliseconds."""
    runs = [await _one(pipeline, clip, clock) for clip in clips]
    edits = words = 0
    for run in runs:
        wrong, total = word_errors(run.clip.reference_text, run.heard)
        edits, words = edits + wrong, words + total
    timed = [r for r in runs if r.total_ms is not None]
    totals = [r.total_ms for r in timed if r.total_ms is not None]
    with_terms = [
        r
        for r in runs
        if matched_entries(r.clip.reference_text, glossary.for_direction(r.clip.src, r.clip.tgt))
    ]
    with_digits = [r for r in runs if any(c.isdigit() for c in r.clip.reference_text)]
    report = HarnessReport(
        n=len(runs),
        n_silent=sum(not r.events for r in runs),
        stage_ms={
            "stt": _mean([r.stt_ms for r in timed if r.stt_ms is not None]) if timed else 0.0,
            "translate": _mean([r.translate_ms for r in timed if r.translate_ms is not None])
            if timed
            else 0.0,
            "tts_first": _mean([r.tts_first_ms for r in timed if r.tts_first_ms is not None])
            if timed
            else 0.0,
        },
        p50_ms=percentile(totals, 50) if totals else None,
        p95_ms=percentile(totals, 95) if totals else None,
        wer=edits / words,
        chrf=chrf(
            [r.clip.reference_translation for r in runs],
            [r.final.tgt_text if r.final else "" for r in runs],
        ),
        glossary_acc=(
            _mean([float(r.final is not None and not r.final.glossary_missing) for r in with_terms])
            if with_terms
            else None
        ),
        digit_acc=(
            _mean([float(r.final is not None and r.final.digits_ok) for r in with_digits])
            if with_digits
            else None
        ),
    )
    return HarnessResult(report=report, runs=tuple(runs))


def record_replay(runs: Sequence[ClipRun], out: Path) -> ReplayManifest:
    """Lay the measured turns on one timeline (original speech, then the measured wait, then the
    translated speech) and write four Ogg tracks and the manifest the API serves."""
    tracks: dict[tuple[Role, str], list[tuple[int, np.ndarray]]] = {}
    events: list[CaptionEvent] = []
    totals: list[int] = []
    cursor = START_MS

    def place(role: Role, lang: str, at_ms: int, samples: np.ndarray) -> None:
        tracks.setdefault((role, lang), []).append((at_ms, samples))

    for run in runs:
        final, wait = run.final, run.total_ms
        if final is None or wait is None:
            continue  # a turn nobody heard is not part of the demo
        clip = run.clip
        spoken = to_16k(clip.utterance.pcm, clip.utterance.sample_rate)
        chunks = [e for e in run.events if isinstance(e, AudioChunk)]
        translated = np.concatenate([to_16k(c.pcm, c.sample_rate) for c in chunks])
        speech_end = cursor + round(len(spoken) / RATE * 1000)
        first_audio = speech_end + round(wait)
        place(clip.utterance.speaker, clip.src, cursor, spoken)
        place(clip.utterance.speaker, clip.tgt, first_audio, translated)
        events.append(
            final.model_copy(
                update={"t_speech_end_ms": speech_end, "t_first_audio_ms": first_audio}
            )
        )
        totals.append(first_audio - speech_end)
        cursor = first_audio + round(len(translated) / RATE * 1000) + GAP_MS

    length = round(cursor / 1000 * RATE)
    listed: list[ReplayTrack] = []
    for (role, lang), pieces in sorted(tracks.items()):
        timeline = np.zeros(length, dtype=np.float32)
        for at_ms, samples in pieces:
            start = round(at_ms / 1000 * RATE)
            timeline[start : start + len(samples)] = samples
        name = f"audio/{role}-{lang}.ogg"
        write_ogg(out / name, timeline)
        listed.append(ReplayTrack(role=role, lang=lang, url=name))
    manifest = ReplayManifest(
        replay_id=out.name,
        tracks=tuple(listed),
        events=tuple(events),
        latency_p50_ms=round(percentile(totals, 50)),
        latency_p95_ms=round(percentile(totals, 95)),
        n=len(events),
    )
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest
