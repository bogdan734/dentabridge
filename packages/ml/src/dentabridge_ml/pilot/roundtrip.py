"""ML1: can a LoRA-tuned Qwen3-VL be served by Ollama, or must it go through MLX/Transformers?

The pilot tunes the untuned base for a few steps, merges, imports the result into Ollama (two
routes) and compares its answers with Transformers' greedy decoding on the same images. This
module holds the report and the rule that turns reports into a serving decision.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
