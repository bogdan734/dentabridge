"""Training configuration. One file per hardware route lives in packages/ml/configs/."""

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "train"


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_id: str  # a local directory or a Hugging Face id
    image_max_px: int = Field(gt=0)
    lora_r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(default=0.0, ge=0, lt=1)
    train_vision: bool
    quant: Literal["none", "nf4"]
    dtype: Literal["fp16", "bf16"]
    lr: float = Field(gt=0)
    epochs: float = Field(gt=0)
    batch_size: int = Field(default=1, gt=0)
    grad_accum: int = Field(gt=0)
    max_seq_len: int = Field(default=2048, gt=0)
    gradient_checkpointing: bool = False  # trades speed for memory: on for 16 GB cards
    max_steps: int | None = Field(default=None, gt=0)  # set for pilots; overrides epochs
    save_every: int = Field(gt=0)
    push_to: str | None = None  # a private Hub repo for checkpoints, never public
    seed: int = 0
    device: Literal["cuda", "mps"] = "cuda"

    @model_validator(mode="after")
    def _hardware_rules(self) -> Self:
        if self.device == "mps" and self.dtype == "bf16":
            raise ValueError("fp16 on MPS: set dtype = fp16")
        return self


def load_config(path: Path) -> TrainConfig:
    return TrainConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
