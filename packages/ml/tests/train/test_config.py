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


def test_smoke_changes_only_the_length() -> None:
    lean = load_config(CONFIG_DIR / "lean-t4.toml").model_dump()
    smoke = load_config(CONFIG_DIR / "smoke-t4.toml").model_dump()
    assert {k for k in lean if lean[k] != smoke[k]} == {"epochs", "max_steps", "save_every"}
    assert smoke["max_steps"] == 10


def test_every_shipped_config_loads() -> None:
    paths = sorted(CONFIG_DIR.glob("*.toml"))
    assert len(paths) >= 2
    assert all(isinstance(load_config(p), TrainConfig) for p in paths)


@pytest.mark.parametrize(
    "change",
    [
        {"device": "rocm"},
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
