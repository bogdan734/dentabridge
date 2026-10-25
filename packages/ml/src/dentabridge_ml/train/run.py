"""LoRA fine-tuning of Qwen3-VL on the SFT rows: one code path for CUDA, ROCm and MPS.

Needs the `train` extra. Nothing here runs in CI; the tiny-model test (marked slow) covers it.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments

from dentabridge_ml.provenance import git_commit
from dentabridge_ml.train.config import TrainConfig
from dentabridge_ml.train.data import SftCollator, SftRow, read_sft
from dentabridge_ml.train.lora import lora_config
from dentabridge_ml.train.record import make_record

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


class Rows(Dataset):  # type: ignore[misc]  # torch is Any for mypy, see pyproject.toml
    """Rows stay plain objects; the collator turns a batch of them into tensors."""

    def __init__(self, rows: Sequence[SftRow]) -> None:
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> SftRow:
        return self._rows[index]


def load_processor(base_id: str) -> Any:
    processor = AutoProcessor.from_pretrained(base_id)
    processor.tokenizer.padding_side = "right"  # the loss mask counts from the left
    return processor


def attach_lora(model: Any, cfg: TrainConfig) -> Any:
    if cfg.quant == "nf4":
        from peft import prepare_model_for_kbit_training  # needs bitsandbytes (CUDA only)

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return get_peft_model(model, lora_config(cfg))


def load_model(cfg: TrainConfig) -> Any:
    kwargs: dict[str, Any] = {"dtype": DTYPES[cfg.dtype]}
    if cfg.quant == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPES[cfg.dtype],
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    return attach_lora(Qwen3VLForConditionalGeneration.from_pretrained(cfg.base_id, **kwargs), cfg)


def make_trainer(
    model: Any,
    processor: Any,
    cfg: TrainConfig,
    rows: Sequence[SftRow],
    root: Path,
    out: Path,
    amp: bool = True,
) -> Trainer:
    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps or -1,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=amp and cfg.dtype == "bf16",
        fp16=amp and cfg.dtype == "fp16",
        logging_steps=5,
        save_strategy="steps",
        save_steps=cfg.save_every,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=[],
        seed=cfg.seed,
        dataloader_num_workers=0,
        label_names=["labels"],
        push_to_hub=cfg.push_to is not None,
        hub_model_id=cfg.push_to,
        hub_private_repo=True,
        hub_strategy="checkpoint",
        use_cpu=not amp and not torch.cuda.is_available() and not torch.backends.mps.is_available(),
    )
    collator = SftCollator(processor, root, cfg.image_max_px, cfg.max_seq_len)
    return Trainer(model=model, args=args, train_dataset=Rows(rows), data_collator=collator)


def _hardware() -> str:
    if torch.cuda.is_available():
        return str(torch.cuda.get_device_name(0))
    return "Apple MPS" if torch.backends.mps.is_available() else "cpu"


def train(cfg: TrainConfig, train_file: Path, root: Path, out: Path, resume: bool = False) -> Path:
    """Run a configured fine-tune and return the directory holding the LoRA adapter."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    rows = read_sft(train_file)
    trainer = make_trainer(
        load_model(cfg),
        load_processor(cfg.base_id),
        cfg,
        rows,
        root,
        out,
        amp=cfg.device
        != "mps",  # Trainer's mixed precision is CUDA/ROCm only; MPS loads fp16 weights
    )
    has_checkpoint = any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume and has_checkpoint else None)
    adapter = out / "adapter"
    trainer.save_model(str(adapter))
    history = trainer.state.log_history
    (out / "log_history.json").write_text(json.dumps(history), encoding="utf-8")
    record = make_record(cfg, len(rows), history, git_commit(), _hardware())
    (out / "run.json").write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return adapter
