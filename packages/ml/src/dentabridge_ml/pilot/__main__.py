"""`python -m dentabridge_ml.pilot gguf|compare`: ML1 by hand. Needs Ollama and the train extra.

Ollama (0.40.10) cannot load a Qwen3-VL safetensors directory: it imports those as MLX models and
its MLX runner has no Qwen3-VL. The route that works is llama.cpp's converter: a quantised
language-model GGUF plus an f16 projector GGUF, registered with a two-FROM Modelfile. `gguf` does
that for a merged (or base) checkpoint directory. `compare` answers the validation photos of a
reference run (`eval run --backend transformers:<dir> --limit N`) with Ollama and writes the
round-trip report with the serving verdict.
"""

import argparse
import asyncio
import base64
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from dentabridge_core import VisionBackend, VisionOutput
from dentabridge_ml.data.loaders import load_records
from dentabridge_ml.data.sources import default_root
from dentabridge_ml.eval.predict import read_predictions, research_release, sanitise
from dentabridge_ml.pilot.roundtrip import (
    COLOURS,
    RoundtripReport,
    agreement,
    colour_probe,
    serving_decision,
    valid_rate,
)
from dentabridge_ml.serving import OLLAMA_URL
from dentabridge_ml.serving.ollama import OllamaVisionBackend
from dentabridge_ml.serving.parse import ParseError
from dentabridge_ml.splits.build import MANIFEST_DIR
from dentabridge_ml.splits.manifest import read_entries

CONVERTER = "convert_hf_to_gguf.py"


def modelfile(text: Path, projector: Path) -> str:
    """The Modelfile for a language-model GGUF and its vision projector GGUF.

    The stock `qwen3-vl:4b` is the thinking variant; ours is tuned from the Instruct checkpoint,
    so it gets the instruct renderer and parser and the Instruct sampling defaults.
    """
    return (
        f"FROM {text.resolve()}\n"
        f"FROM {projector.resolve()}\n"
        "TEMPLATE {{ .Prompt }}\n"
        "RENDERER qwen3-vl-instruct\n"
        "PARSER qwen3-vl-instruct\n"
        "PARAMETER top_k 20\n"
        "PARAMETER top_p 0.8\n"
    )


def _convert(converter: Path, checkpoint: Path, out: Path, *flags: str) -> None:
    # sentencepiece is imported by the converter's Qwen vocabulary path before it falls back to BPE
    subprocess.run(
        [sys.executable, str(converter), str(checkpoint), "--outfile", str(out), *flags], check=True
    )


def _gguf(args: argparse.Namespace) -> int:
    converter = args.llama_cpp / CONVERTER
    if not converter.exists():
        raise FileNotFoundError(f"{converter} is missing: run scripts/llama-cpp.sh")
    args.out.mkdir(parents=True, exist_ok=True)
    text, projector = args.out / f"{args.name}-{args.outtype}.gguf", args.out / "mmproj-f16.gguf"
    _convert(converter, args.checkpoint, text, "--outtype", args.outtype)
    _convert(converter, args.checkpoint, projector, "--mmproj", "--outtype", "f16")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Modelfile"
        path.write_text(modelfile(text, projector), encoding="utf-8")
        subprocess.run(["ollama", "create", args.name, "-f", str(path)], check=True)
    return 0


def _chat(model: str, prompt: str, image: bytes, **extra: Any) -> dict[str, Any]:
    body = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 8},
        "messages": [
            {"role": "user", "content": prompt, "images": [base64.b64encode(image).decode()]}
        ],
        **extra,
    }
    response = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=600.0)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


def sees_images(model: str) -> bool:
    """Ollama names the colour of a big square: the image reached the language model intact."""
    ask = "What colour is the big square? Answer with one word."
    return all(
        name in _chat(model, ask, colour_probe(name))["message"]["content"].lower()
        for name in COLOURS
    )


def has_logprobs(model: str) -> bool:
    data = _chat(model, "Say ok.", colour_probe("red"), logprobs=True, top_logprobs=2)
    return bool(data.get("logprobs"))


async def _answers(backend: VisionBackend, images: Sequence[Any]) -> list[VisionOutput | None]:
    answers: list[VisionOutput | None] = []
    for image in images:
        try:
            answers.append(await backend.analyse(image, "intraoral_photo"))
        except ParseError:
            answers.append(None)
    return answers


def _compare(args: argparse.Namespace) -> int:
    reference = read_predictions(args.reference)
    ids = [row.record_id for row in reference]
    entries = {e.record_id: e for e in read_entries(MANIFEST_DIR, args.split)}
    sources = sorted({entries[i].source for i in ids})
    records = {r.record_id: r for s in sources for r in load_records(s, args.root)}
    images = [sanitise(args.root / records[i].source / records[i].path) for i in ids]
    expected = [row.output for row in reference]

    def served(constrain: bool) -> OllamaVisionBackend:
        return OllamaVisionBackend(
            OLLAMA_URL,
            args.name,
            research_release(sources),
            image_max_px=args.image_max_px,
            constrain=constrain,
        )

    # free decoding runs the same computation as the reference, so it measures the import; the
    # schema-constrained run is how the model is served, so it gives the valid-JSON rate
    free = asyncio.run(_answers(served(constrain=False), images))
    constrained = asyncio.run(_answers(served(constrain=True), images))

    version = httpx.get(f"{OLLAMA_URL}/api/version", timeout=10.0).json()["version"]
    report = RoundtripReport(
        path=args.route,
        vision_ok=sees_images(args.name),
        json_valid_rate=valid_rate(constrained),
        agreement=agreement(expected, free),
        logprobs_available=has_logprobs(args.name),
        ollama_version=version,
    )
    report.write(args.out)
    print(report.model_dump_json(indent=2))
    verdict = serving_decision([report])
    print(f"reference valid: {valid_rate(expected):.2f}; serving verdict: {verdict}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dentabridge_ml.pilot")
    steps = parser.add_subparsers(dest="step", required=True)

    gguf = steps.add_parser("gguf", help="convert a checkpoint directory and register it in Ollama")
    gguf.add_argument("checkpoint", type=Path, help="a merged (or base) Hugging Face directory")
    gguf.add_argument("--name", required=True, help="the Ollama model name")
    gguf.add_argument("--out", type=Path, required=True, help="where the GGUF files go")
    gguf.add_argument("--outtype", choices=("q8_0", "f16"), default="q8_0")
    gguf.add_argument("--llama-cpp", type=Path, default=Path("var/tools/llama.cpp"))
    gguf.set_defaults(func=_gguf)

    compare = steps.add_parser("compare", help="Ollama vs a Transformers reference run")
    compare.add_argument("reference", type=Path, help="predictions.jsonl from `eval run`")
    compare.add_argument("--name", required=True, help="the model name given to `gguf`")
    compare.add_argument("--split", choices=("val",), default="val")
    compare.add_argument("--route", choices=("safetensors", "gguf_mmproj"), default="safetensors")
    compare.add_argument("--image-max-px", type=int, required=True)
    compare.add_argument("--root", type=Path, default=default_root())
    compare.add_argument("--out", type=Path, required=True)
    compare.set_defaults(func=_compare)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
