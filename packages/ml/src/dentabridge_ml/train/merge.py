"""Fold a LoRA adapter into the base weights: the one artefact Ollama, MLX and the Hub take."""

import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

from dentabridge_ml.train.run import DTYPES

TEXT_FILES = ["*.json", "*.txt", "*.jinja"]
WEIGHT_INDEX = "model.safetensors.index.json"


def _checkpoint_files(base_id: str) -> list[Path]:
    """The base checkpoint's own config, tokenizer and processor files, as published."""
    folder = Path(base_id)
    if not folder.is_dir():
        folder = Path(snapshot_download(base_id, allow_patterns=TEXT_FILES))
    return [p for p in folder.iterdir() if p.is_file() and p.name != WEIGHT_INDEX]


def merge_adapter(base_id: str, adapter: Path, out: Path, dtype: str = "bf16") -> Path:
    """Write base + adapter as a plain Hugging Face directory.

    The weights are new; every other file is copied from the base checkpoint, because converters
    (Ollama's among them) read the files the publisher wrote, not what a newer transformers
    would regenerate.
    """
    base: Any = Qwen3VLForConditionalGeneration.from_pretrained(base_id, dtype=DTYPES[dtype])
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    with torch.no_grad():
        merged.save_pretrained(out, safe_serialization=True)
    for source in _checkpoint_files(base_id):
        if source.suffix != ".safetensors":
            shutil.copy2(source, out / source.name)
    return out
