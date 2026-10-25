"""The training path on a tiny random Qwen3-VL with the real tokenizer and processor.

Needs the `train` extra and the small files of Qwen3-VL-4B-Instruct (tokenizer, processor,
config; not the weights), so it skips itself in CI. Run with `-m slow`.
"""

import math
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from transformers import AutoConfig, Qwen3VLForConditionalGeneration  # noqa: E402

from dentabridge_core import FindingCode  # noqa: E402
from dentabridge_ml.data.sources import default_root  # noqa: E402
from dentabridge_ml.targets.prompt import build_prompt  # noqa: E402
from dentabridge_ml.train.config import TrainConfig  # noqa: E402
from dentabridge_ml.train.data import END_OF_TURN, SftCollator, SftRow  # noqa: E402
from dentabridge_ml.train.run import attach_lora, load_processor, make_trainer  # noqa: E402

MODEL_DIR = default_root() / "models" / "Qwen3-VL-4B-Instruct"
IMAGE_TOKEN = 151655
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (MODEL_DIR / "tokenizer.json").exists(), reason="Qwen files not downloaded"
    ),
]


def _config(train_vision: bool = False) -> TrainConfig:
    return TrainConfig(
        base_id=str(MODEL_DIR),
        image_max_px=64,
        lora_r=4,
        lora_alpha=8,
        train_vision=train_vision,
        quant="none",
        dtype="bf16",
        lr=1e-3,
        epochs=1,
        max_steps=2,
        batch_size=2,
        grad_accum=1,
        max_seq_len=1024,
        save_every=2,
    )


def _tiny_model() -> Any:
    config = AutoConfig.from_pretrained(MODEL_DIR)
    text, vision = config.text_config, config.vision_config
    text.hidden_size, text.intermediate_size = 64, 128
    text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads = 2, 4, 2
    text.head_dim = 16
    text.rope_parameters = {
        **text.rope_parameters,
        "mrope_section": [4, 2, 2],
    }  # sums to head_dim / 2
    vision.depth, vision.hidden_size, vision.intermediate_size, vision.num_heads = 2, 64, 128, 4
    vision.out_hidden_size, vision.deepstack_visual_indexes = 64, [0, 1]
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).float()


def _rows(tmp_path: Path) -> list[SftRow]:
    rows = []
    for i in range(4):
        path = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (96, 64), (40 * i, 90, 120)).save(path)
        answer = '{"image_quality":"ok","findings":[]}' + "x" * (i * 5)
        prompt = build_prompt(frozenset({FindingCode.CARIES}), "intraoral_photo")
        rows.append(SftRow(id=f"r{i}", image=path.name, prompt=prompt, answer=answer))
    return rows


def test_collate_masks_prompt_and_image_tokens(tmp_path: Path) -> None:
    processor = load_processor(str(MODEL_DIR))
    rows = _rows(tmp_path)[:2]
    batch = SftCollator(processor, tmp_path, 64, 1024)(rows)
    ids, labels, mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
    assert (ids == IMAGE_TOKEN).any(), "the prompt really contains image tokens"
    assert not (labels == IMAGE_TOKEN).any(), "image tokens are never trained on"
    assert (labels[mask == 0] == -100).all(), "padding is ignored"
    for i, row in enumerate(rows):
        kept = labels[i][labels[i] != -100]
        assert processor.tokenizer.decode(kept) == row.answer + END_OF_TURN
    assert "mm_token_type_ids" in batch
    assert rows[0].answer != rows[1].answer


def test_collator_refuses_to_truncate(tmp_path: Path) -> None:
    processor = load_processor(str(MODEL_DIR))
    with pytest.raises(ValueError, match="limit 10"):
        SftCollator(processor, tmp_path, 64, 10)(_rows(tmp_path)[:1])


def test_lora_trains_language_layers_only() -> None:
    wrapped = attach_lora(_tiny_model(), _config())
    names = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert names
    assert all("lora_" in n and "language_model" in n for n in names)
    assert not any("visual" in n for n in names)
    with_vision = attach_lora(_tiny_model(), _config(train_vision=True))
    assert any("visual" in n for n, p in with_vision.named_parameters() if p.requires_grad)


def test_two_steps_tiny_model(tmp_path: Path) -> None:
    processor = load_processor(str(MODEL_DIR))
    model = attach_lora(_tiny_model(), _config())
    trainer = make_trainer(
        model, processor, _config(), _rows(tmp_path), tmp_path, tmp_path / "out", amp=False
    )
    trainer.train()
    assert trainer.state.global_step == 2
    mean_loss = trainer.state.log_history[-1]["train_loss"]
    assert math.isfinite(mean_loss), "a random model starts near ln(vocab), not at nan or inf"
    trainer.save_model(str(tmp_path / "adapter"))
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
