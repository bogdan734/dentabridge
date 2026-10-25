from pathlib import Path

import pytest
from pydantic import ValidationError

from dentabridge_ml.train.config import CONFIG_DIR, TrainConfig, load_config


def test_lean_config_is_t4_safe() -> None:
    cfg = load_config(CONFIG_DIR / "lean-t4.toml")
    assert (cfg.device, cfg.quant, cfg.dtype) == ("cuda", "nf4", "fp16")  # T4 has no bf16
    assert cfg.image_max_px <= 448
    assert cfg.batch_size * cfg.grad_accum >= 8
    assert cfg.save_every <= 500, "a 12 h Kaggle session can end at any time"
    assert cfg.push_to is None
    assert cfg.gradient_checkpointing, "16 GB needs it"


def test_rocm_configs_have_no_bitsandbytes() -> None:
    for name in ("mi300x-base.toml", "pilot-50.toml"):
        cfg = load_config(CONFIG_DIR / name)
        assert (cfg.device, cfg.quant, cfg.dtype) == ("rocm", "none", "bf16")


def test_pilot_changes_only_the_length() -> None:
    base = load_config(CONFIG_DIR / "mi300x-base.toml").model_dump()
    pilot = load_config(CONFIG_DIR / "pilot-50.toml").model_dump()
    differing = {k for k in base if base[k] != pilot[k]}
    assert differing == {"epochs", "max_steps", "save_every"}
    assert pilot["max_steps"] == 50


def test_every_shipped_config_loads() -> None:
    paths = sorted(CONFIG_DIR.glob("*.toml"))
    assert len(paths) >= 3
    assert all(isinstance(load_config(p), TrainConfig) for p in paths)


@pytest.mark.parametrize(
    "change",
    [
        {"device": "rocm", "quant": "nf4"},
        {"device": "mps", "dtype": "bf16"},
        {"lora_r": 0},
        {"unknown_field": 1},
        {"push_to": 5},
    ],
)
def test_invalid_configs_are_rejected(change: dict[str, object], tmp_path: Path) -> None:
    data = load_config(CONFIG_DIR / "lean-t4.toml").model_dump()
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({**data, **change})


@pytest.mark.parametrize(
    ("name", "changed"),
    [
        ("abl-vision", {"train_vision"}),
        ("abl-672px", {"image_max_px"}),
        ("abl-1024px", {"image_max_px"}),
        ("abl-r64", {"lora_r", "lora_alpha"}),  # alpha follows r so the scale stays 1
    ],
)
def test_each_ablation_changes_one_thing_from_the_base_recipe(name: str, changed: set[str]) -> None:
    base = load_config(CONFIG_DIR / "mi300x-base.toml").model_dump()
    variant = load_config(CONFIG_DIR / f"{name}.toml").model_dump()
    assert {k for k in base if base[k] != variant[k]} == changed
