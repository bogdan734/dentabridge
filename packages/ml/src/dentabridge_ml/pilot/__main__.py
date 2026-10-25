"""`python -m dentabridge_ml.pilot import|compare`: ML1 by hand. Needs Ollama and the train extra.

`import` registers a merged model in Ollama from its safetensors directory, copying the template
and parameters of a stock model. `compare` answers the same validation photos with Transformers
(the reference) and with Ollama, and writes the round-trip report with the serving verdict.
"""

import argparse
import asyncio
import base64
import random
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from dentabridge_core import VisionBackend, VisionOutput
from dentabridge_ml.data.loaders import load_records
from dentabridge_ml.data.sources import default_root
from dentabridge_ml.eval.predict import research_release, sanitise
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


def modelfile(stock: str, merged: Path) -> str:
    """The stock model's Modelfile (template, renderer, parameters), FROM pointing at `merged`."""
    shown = subprocess.run(
        ["ollama", "show", stock, "--modelfile"], capture_output=True, text=True, check=True
    ).stdout
    kept = [
        line for line in shown.splitlines() if line.strip() and not line.startswith(("#", "FROM"))
    ]
    return "\n".join([f"FROM {merged.resolve()}", *kept]) + "\n"


def _import(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Modelfile"
        path.write_text(modelfile(args.like, args.merged), encoding="utf-8")
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
    import torch

    from dentabridge_ml.serving.transformers_backend import TransformersVisionBackend

    entries = read_entries(MANIFEST_DIR, "val")
    chosen = random.Random(args.seed).sample(list(entries), args.n)
    records = {
        r.record_id: r
        for s in sorted({e.source for e in chosen})
        for r in load_records(s, args.root)
    }
    images = [
        sanitise(args.root / records[e.record_id].source / records[e.record_id].path)
        for e in chosen
    ]

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    reference = TransformersVisionBackend(
        str(args.merged), research_release(), args.image_max_px, dtype
    )
    expected = asyncio.run(_answers(reference, images))
    del reference
    ollama = OllamaVisionBackend(
        OLLAMA_URL, args.name, research_release(), image_max_px=args.image_max_px
    )
    got = asyncio.run(_answers(ollama, images))

    version = httpx.get(f"{OLLAMA_URL}/api/version", timeout=10.0).json()["version"]
    report = RoundtripReport(
        path=args.route,
        vision_ok=sees_images(args.name),
        json_valid_rate=valid_rate(got),
        agreement=agreement(expected, got),
        logprobs_available=has_logprobs(args.name),
        ollama_version=version,
    )
    report.write(args.out)
    print(report.model_dump_json(indent=2))
    verdict = serving_decision([report])
    print(f"transformers valid: {valid_rate(expected):.2f}; serving verdict: {verdict}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dentabridge_ml.pilot")
    steps = parser.add_subparsers(dest="step", required=True)

    register = steps.add_parser(
        "import", help="create an Ollama model from a merged safetensors dir"
    )
    register.add_argument("merged", type=Path)
    register.add_argument("--name", required=True)
    register.add_argument(
        "--like", default="qwen3-vl:4b", help="stock model to copy the template from"
    )
    register.set_defaults(func=_import)

    compare = steps.add_parser("compare", help="Transformers vs Ollama on the same photos")
    compare.add_argument("merged", type=Path)
    compare.add_argument("--name", required=True, help="the model name given to `import`")
    compare.add_argument("--route", choices=("safetensors", "gguf_mmproj"), default="safetensors")
    compare.add_argument("--n", type=int, default=20)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--image-max-px", type=int, required=True)
    compare.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    compare.add_argument("--root", type=Path, default=default_root())
    compare.add_argument("--out", type=Path, required=True)
    compare.set_defaults(func=_compare)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
