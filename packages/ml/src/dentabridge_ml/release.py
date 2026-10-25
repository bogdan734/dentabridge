"""The release gate: which findings the shipped model may report as assessed.

The thresholds live in configs/release_gate.toml and are committed BEFORE the test split is
evaluated for the first time, so the bar cannot move after the result is known.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dentabridge_ml.eval.results import CodeResult, SplitResult
from dentabridge_ml.serving.parse import ReleaseConfig


class Gate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_positives: int = Field(gt=0)
    min_sens_lower: float = Field(ge=0, le=1)  # lower bound of the sensitivity interval
    min_spec_lower: float = Field(ge=0, le=1)  # lower bound of the specificity interval
    max_ece: float = Field(ge=0, le=1)

    @classmethod
    def read(cls, path: Path) -> "Gate":
        return cls.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def passes(result: CodeResult, gate: Gate) -> bool:
    """Every number must exist and clear its bar; a missing side (no negatives) fails."""
    if result.n_pos < gate.min_positives or result.sens_ci is None or result.spec_ci is None:
        return False
    return (
        result.sens_ci[0] >= gate.min_sens_lower
        and result.spec_ci[0] >= gate.min_spec_lower
        and result.ece is not None
        and result.ece <= gate.max_ece
    )


def apply_gate(test: SplitResult, gate: Gate) -> ReleaseConfig:
    return ReleaseConfig(
        assessed=frozenset(c.code for c in test.codes if passes(c, gate)),
        needs_scores=False,
    )
