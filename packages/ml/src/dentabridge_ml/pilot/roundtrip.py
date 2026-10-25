"""ML1: can a LoRA-tuned Qwen3-VL be served by Ollama, or must it go through MLX/Transformers?

The pilot tunes the untuned base for a few steps, merges, imports the result into Ollama (two
routes) and compares its answers with Transformers' greedy decoding on the same images. This
module holds the report and the rule that turns reports into a serving decision.
"""

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from dentabridge_core import VisionOutput

MIN_AGREEMENT = 0.90
MIN_JSON_VALID = 0.95

ImportPath = Literal["safetensors", "gguf_mmproj"]
Serving = Literal["ollama", "mlx_or_transformers"]


class RoundtripReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: ImportPath
    vision_ok: bool  # Ollama answered about the image, not about nothing
    json_valid_rate: float = Field(ge=0, le=1)
    agreement: float = Field(ge=0, le=1)  # same parsed answer as Transformers greedy decoding
    logprobs_available: bool
    ollama_version: str

    def passes(self) -> bool:
        return (
            self.vision_ok
            and self.agreement >= MIN_AGREEMENT
            and self.json_valid_rate >= MIN_JSON_VALID
        )

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


def serving_decision(reports: Sequence[RoundtripReport]) -> Serving:
    """Ollama serves the tuned model only if some import route passes every bar."""
    return "ollama" if any(r.passes() for r in reports) else "mlx_or_transformers"


def _gist(output: VisionOutput) -> tuple[object, ...]:
    findings = tuple((f.code.value, f.status) for f in output.findings)
    return (output.image_quality, output.quality_reason, findings)


def same_answer(reference: VisionOutput | None, other: VisionOutput | None) -> bool:
    """Same image quality and the same status for every code. Boxes are not compared: they move
    by a pixel with any numerical difference and carry no decision. Two answers that both failed
    to parse prove nothing, so they do not agree."""
    if reference is None or other is None:
        return False
    return _gist(reference) == _gist(other)


def agreement(
    reference: Sequence[VisionOutput | None], other: Sequence[VisionOutput | None]
) -> float:
    if len(reference) != len(other):
        raise ValueError("both runs need the same number of images")
    return sum(same_answer(a, b) for a, b in zip(reference, other, strict=True)) / len(reference)


def valid_rate(outputs: Sequence[VisionOutput | None]) -> float:
    return sum(o is not None for o in outputs) / len(outputs)


COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 60, 220),
}


def colour_probe(name: str) -> bytes:
    """A big square of one colour on grey: if Ollama names it, the image reached the model."""
    image = Image.new("RGB", (256, 256), (128, 128, 128))
    ImageDraw.Draw(image).rectangle((48, 48, 208, 208), fill=COLOURS[name])
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95)
    return buffer.getvalue()
