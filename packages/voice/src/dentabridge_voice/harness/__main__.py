"""`python -m dentabridge_voice.harness run`: measure the pipeline on this machine.

Needs the three services running (Ollama with the translator, a Piper server with the two voices)
and the `stt` and `harness` extras. The scripted turns are spoken by the Piper voices, so the
recognition numbers are for synthetic speech; the result files say so.
"""

import argparse
import asyncio
import os
import platform
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from dentabridge_core import Role
from dentabridge_voice.adapters.faster_whisper import FasterWhisperSTT
from dentabridge_voice.adapters.ollama_translator import OllamaTranslator
from dentabridge_voice.adapters.piper_http import VOICES, PiperHttpTTS
from dentabridge_voice.glossary import GLOSSARY_DIR, load_glossary
from dentabridge_voice.harness.audio import RATE, to_16k, to_pcm16
from dentabridge_voice.harness.results import VoiceResults, render_markdown
from dentabridge_voice.harness.run import Clip, load_script, record_replay, run_harness
from dentabridge_voice.pipeline import Pipeline
from dentabridge_voice.ports import Utterance

WARM_UP = 2


class WallMs:
    def now_ms(self) -> int:
        return round(time.perf_counter() * 1000)


async def _clips(tts: PiperHttpTTS, limit: int | None) -> list[Clip]:
    clips = []
    for i, line in enumerate(load_script()[:limit]):
        pcm = b"".join([chunk async for chunk in tts.synthesise(line.text, line.lang)])
        speaker: Role = line.speaker
        other = "ru" if line.lang == "en" else "en"
        clips.append(
            Clip(
                utterance=Utterance(
                    utterance_id=f"u{i:02d}",
                    speaker=speaker,
                    pcm=to_pcm16(to_16k(pcm, tts.sample_rate)),
                    sample_rate=RATE,
                    t_speech_end_ms=0,
                ),
                src=line.lang,
                tgt=other,
                reference_text=line.text,
                reference_translation=line.translation,
            )
        )
    return clips


async def _run(args: argparse.Namespace) -> None:
    glossary = load_glossary(GLOSSARY_DIR / "en-ru.tsv")
    tts = PiperHttpTTS(args.piper, VOICES)
    translator = OllamaTranslator(args.ollama, args.translator)
    stt = FasterWhisperSTT(args.whisper, args.device, args.compute_type)
    pipeline = Pipeline(stt, translator, tts, glossary, WallMs())
    clips = await _clips(tts, args.limit)
    await run_harness(pipeline, clips[:WARM_UP], glossary)  # load the models before timing
    load_average = os.getloadavg()[0]
    result = await run_harness(pipeline, clips, glossary)
    results = VoiceResults(
        profile=args.profile,
        recorded_on=date.today(),
        hardware=args.hardware or platform.platform(),
        load_average=load_average,
        audio="synthetic speech from Piper voices, not human recordings",
        models={
            "speech to text": f"faster-whisper {args.whisper} ({args.compute_type}, {args.device})",
            "translator": f"{args.translator} through Ollama",
            "voices": ", ".join(VOICES.values()) + " through Piper",
        },
        report=result.report,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "voice.json").write_text(results.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (args.out / "voice.md").write_text(render_markdown(results), encoding="utf-8")
    if args.replay:
        manifest = record_replay(result.runs, args.replay)
        print(f"replay: {manifest.n} turns -> {args.replay}")
    print(render_markdown(results))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dentabridge_voice.harness", description=__doc__
    )
    steps = parser.add_subparsers(dest="step", required=True)
    run = steps.add_parser("run", help="measure latency and quality, write voice.json and voice.md")
    run.add_argument("--out", type=Path, required=True, help="docs/results")
    run.add_argument("--replay", type=Path, help="also record the demo replay into this folder")
    run.add_argument("--profile", default="mac-native")
    run.add_argument("--hardware", help='for the report, e.g. "Apple M3 Max, 36 GB"')
    run.add_argument("--ollama", default="http://localhost:11434")
    run.add_argument("--translator", default="qwen3:4b-instruct")
    run.add_argument("--piper", default="http://127.0.0.1:5100")
    run.add_argument("--whisper", default="small")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--compute-type", default="int8")
    run.add_argument("--limit", type=int, help="only the first N turns (a quick trial)")
    run.set_defaults(func=lambda args: asyncio.run(_run(args)))
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
