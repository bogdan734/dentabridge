"""Fold a LoRA adapter into the base weights: the one artefact Ollama, MLX and the Hub take."""

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

from dentabridge_ml.train.run import DTYPES, load_processor


def merge_adapter(base_id: str, adapter: Path, out: Path, dtype: str = "bf16") -> Path:
    """Write base + adapter as a plain Hugging Face directory (weights, tokenizer, processor)."""
    base: Any = Qwen3VLForConditionalGeneration.from_pretrained(base_id, dtype=DTYPES[dtype])
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    with torch.no_grad():
        merged.save_pretrained(out, safe_serialization=True)
    load_processor(base_id).save_pretrained(out)
    return out
