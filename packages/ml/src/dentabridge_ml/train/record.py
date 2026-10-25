"""One JSON record per training run: what ran, on what, how fast. Costs are added by the owner."""

import hashlib
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from dentabridge_ml.train.config import TrainConfig


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config_sha256: str
    commit: str
    hardware: str
    rows: int
    steps: int
    train_seconds: float
    samples_per_second: float
    final_loss: float | None


def make_record(
    cfg: TrainConfig, rows: int, log_history: Sequence[dict[str, float]], commit: str, hardware: str
) -> RunRecord:
    summary = next((h for h in reversed(log_history) if "train_runtime" in h), None)
    if summary is None:
        raise ValueError("the trainer logged no final summary: the run did not finish")
    losses = [h["loss"] for h in log_history if "loss" in h]
    return RunRecord(
        config_sha256=hashlib.sha256(cfg.model_dump_json().encode()).hexdigest(),
        commit=commit,
        hardware=hardware,
        rows=rows,
        steps=int(summary["step"]),
        train_seconds=summary["train_runtime"],
        samples_per_second=summary["train_samples_per_second"],
        final_loss=losses[-1] if losses else summary.get("train_loss"),
    )
