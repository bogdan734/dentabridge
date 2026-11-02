"""The training path on a tiny random Qwen3-VL with the real tokenizer and processor.

Needs the `train` extra and the small files of Qwen3-VL-4B-Instruct (tokenizer, processor,
config; not the weights), so it skips itself in CI. Run with `-m slow`.
"""

import asyncio
import json
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
from dentabridge_ml.eval.predict import sanitise  # noqa: E402
from dentabridge_ml.serving.parse import ParseError, ReleaseConfig  # noqa: E402
from dentabridge_ml.serving.transformers_backend import TransformersVisionBackend  # noqa: E402
from dentabridge_ml.targets.prompt import build_prompt  # noqa: E402
from dentabridge_ml.train.config import TrainConfig  # noqa: E402
from dentabridge_ml.train.data import END_OF_TURN, SftCollator, SftRow  # noqa: E402
from dentabridge_ml.train.merge import merge_adapter  # noqa: E402
from dentabridge_ml.train.run import (  # noqa: E402
    attach_lora,
    load_model,
    load_processor,
    make_trainer,
)

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


def test_merged_weights_answer_like_the_adapter(tmp_path: Path) -> None:
    processor = load_processor(str(MODEL_DIR))
    base = _tiny_model()
    base.save_pretrained(tmp_path / "base")
    processor.save_pretrained(tmp_path / "base")
    tuned = attach_lora(base, _config())
    with torch.no_grad():  # lora_B starts at zero, which would make the comparison trivial
        for name, param in tuned.named_parameters():
            if "lora_B" in name:
                param.normal_(0.0, 0.05)
    tuned.save_pretrained(tmp_path / "adapter")
    batch = SftCollator(processor, tmp_path, 64, 1024)(_rows(tmp_path)[:2])
    batch.pop("labels")
    tuned.eval()
    with torch.no_grad():
        expected = tuned(**batch).logits
    merged_dir = merge_adapter(
        str(tmp_path / "base"), tmp_path / "adapter", tmp_path / "merged", "fp16"
    )
    merged = Qwen3VLForConditionalGeneration.from_pretrained(merged_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        actual = merged(**batch).logits
    assert (tmp_path / "merged" / "tokenizer.json").exists()
    assert not any("lora" in k for k in merged.state_dict())
    assert torch.allclose(expected, actual, atol=2e-2), (expected - actual).abs().max()


def _backend(tmp_path: Path) -> TransformersVisionBackend:
    processor = load_processor(str(MODEL_DIR))
    _tiny_model().save_pretrained(tmp_path / "base")
    processor.save_pretrained(tmp_path / "base")
    release = ReleaseConfig(assessed=frozenset({FindingCode.CARIES}))
    return TransformersVisionBackend(str(tmp_path / "base"), release, 64, torch.float32, "cpu")


def _photo(tmp_path: Path) -> Any:
    Image.new("RGB", (96, 64), (200, 90, 120)).save(tmp_path / "photo.jpg")
    return sanitise(tmp_path / "photo.jpg")


def test_backend_parses_only_the_newly_generated_tokens_and_scores_each_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    answer = {
        "image_quality": "ok",
        "quality_reason": None,
        "findings": [{"code": "caries", "status": "possible", "regions": []}],
    }
    tokenizer = backend._processor.tokenizer
    text = json.dumps(answer, separators=(",", ":"))
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    possible_id, not_id = backend._ids
    vocab = backend._net.config.text_config.vocab_size
    steps = []
    for token_id in ids["input_ids"][0].tolist():
        row = torch.full((1, vocab), -1e4)
        row[0, token_id] = 0.0
        if token_id == possible_id:  # the model chose "possible" over "not" by one nat
            row[0, token_id], row[0, not_id] = 2.0, 1.0
        steps.append(row)
    seen: dict[str, Any] = {}

    def fake_generate(**inputs: Any) -> Any:
        seen.update(inputs)
        for step in steps:  # what generate() does: show every step's logits to the processors
            for processor in inputs["logits_processor"]:
                processor(None, step)
        return torch.cat([inputs["input_ids"], ids["input_ids"]], dim=1)

    monkeypatch.setattr(backend._net, "generate", fake_generate)
    output = asyncio.run(backend.analyse(_photo(tmp_path), "intraoral_photo"))
    (finding,) = output.findings
    assert (finding.code, finding.status) == (FindingCode.CARIES, "possible")
    assert finding.score == pytest.approx(math.exp(2) / (math.exp(2) + math.exp(1)), abs=1e-4)
    assert seen["do_sample"] is False
    assert seen["max_new_tokens"] == 700
    assert (seen["input_ids"] == IMAGE_TOKEN).any(), "the picture reached the model"


def test_a_batch_of_differently_sized_pictures_is_padded_on_the_left(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    Image.new("RGB", (96, 64), (10, 90, 120)).save(tmp_path / "wide.jpg")
    Image.new("RGB", (40, 120), (200, 90, 20)).save(tmp_path / "tall.jpg")
    images = [sanitise(tmp_path / "wide.jpg"), sanitise(tmp_path / "tall.jpg")]
    results = asyncio.run(backend.analyse_many(images, "intraoral_photo"))
    assert len(results) == 2
    assert all(isinstance(r, ParseError) for r in results), "a random model writes no JSON"
    assert backend._processor.tokenizer.padding_side == "left"


def test_backend_generates_end_to_end_and_reports_unparseable_text(tmp_path: Path) -> None:
    with pytest.raises(ParseError):
        asyncio.run(_backend(tmp_path).analyse(_photo(tmp_path), "intraoral_photo"))


def test_merged_directory_has_the_base_checkpoint_layout(tmp_path: Path) -> None:
    """Ollama and llama.cpp convert by tensor name, shape and dtype: a merge must change none."""
    from safetensors import safe_open

    processor = load_processor(str(MODEL_DIR))
    base = _tiny_model()
    base.save_pretrained(tmp_path / "base")
    processor.save_pretrained(tmp_path / "base")
    tuned = attach_lora(base, _config())
    tuned.save_pretrained(tmp_path / "adapter")
    merged = merge_adapter(
        str(tmp_path / "base"), tmp_path / "adapter", tmp_path / "merged", "fp16"
    )

    def layout(directory: Path) -> dict[str, tuple[tuple[int, ...], str]]:
        found = {}
        for shard in sorted(directory.glob("*.safetensors")):
            with safe_open(shard, "pt") as handle:
                for key in handle.keys():  # noqa: SIM118  (safetensors handles are not mappings)
                    piece = handle.get_slice(key)
                    found[key] = (tuple(piece.get_shape()), piece.get_dtype())
        return found

    base_layout, merged_layout = layout(tmp_path / "base"), layout(merged)
    assert merged_layout.keys() == base_layout.keys()
    assert {k: v[0] for k, v in merged_layout.items()} == {k: v[0] for k, v in base_layout.items()}
    assert not any("lora" in k for k in merged_layout)


def test_merged_directory_keeps_the_original_config_and_processor_files(tmp_path: Path) -> None:
    """Ollama's converter reads the hub checkpoint's own files (preprocessor_config.json...), not
    whatever the installed transformers would write today."""
    processor = load_processor(str(MODEL_DIR))
    base = _tiny_model()
    base.save_pretrained(tmp_path / "base")
    processor.save_pretrained(tmp_path / "base")
    (tmp_path / "base" / "preprocessor_config.json").write_text('{"from": "the hub"}')
    tuned = attach_lora(base, _config())
    tuned.save_pretrained(tmp_path / "adapter")
    merged = merge_adapter(
        str(tmp_path / "base"), tmp_path / "adapter", tmp_path / "merged", "fp16"
    )
    assert (merged / "preprocessor_config.json").read_text() == '{"from": "the hub"}'
    assert (merged / "config.json").read_text() == (tmp_path / "base" / "config.json").read_text()
    assert (merged / "model.safetensors").stat().st_size > 0


@pytest.mark.parametrize("train_vision", [False, True])
def test_two_steps_through_the_nf4_path_with_gradient_checkpointing(
    tmp_path: Path, train_vision: bool
) -> None:
    """The Kaggle recipe's code path (4-bit base, checkpointing, LoRA) on a tiny model, on CPU."""
    bnb = pytest.importorskip("bitsandbytes")
    processor = load_processor(str(MODEL_DIR))
    _tiny_model().save_pretrained(tmp_path / "base")
    processor.save_pretrained(tmp_path / "base")
    cfg = _config().model_copy(
        update={
            "base_id": str(tmp_path / "base"),
            "quant": "nf4",
            "dtype": "bf16",
            "gradient_checkpointing": True,
            "train_vision": train_vision,
        }
    )
    model = load_model(cfg)
    layers = {name: type(module) for name, module in model.named_modules()}
    assert any(t.__name__ == "Linear4bit" for t in layers.values()), "the language model is 4-bit"
    assert not any(
        "visual" in name and isinstance(module, bnb.nn.Linear4bit)
        for name, module in model.named_modules()
    ), "the vision tower stays in 16 bit"
    trainer = make_trainer(
        model, processor, cfg, _rows(tmp_path), tmp_path, tmp_path / "out", amp=False
    )
    trainer.train()
    assert trainer.state.global_step == 2
    assert math.isfinite(trainer.state.log_history[-1]["train_loss"])
    moved = [n for n, p in model.named_parameters() if "lora_B" in n and p.abs().sum() > 0]
    assert moved, "gradients reached the adapters through the checkpointed 4-bit layers"
    assert any("visual" in n for n in moved) == train_vision, "vision adapters learn when asked to"
