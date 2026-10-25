"""Which layers get LoRA adapters."""

from peft import LoraConfig

from dentabridge_ml.train.config import TrainConfig

LANGUAGE = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
VISION = r".*visual\.blocks.*\.(qkv|proj|linear_fc1|linear_fc2)"


def lora_config(cfg: TrainConfig) -> LoraConfig:
    """Language-model layers always; the vision tower only when the config says so.

    A regex string is matched against the full module name by PEFT.
    """
    pattern = f"({LANGUAGE})|({VISION})" if cfg.train_vision else LANGUAGE
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=pattern,
        bias="none",
    )
