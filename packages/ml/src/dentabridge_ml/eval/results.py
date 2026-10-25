"""The shape of an evaluation result. README tables and the model card render from these files."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dentabridge_core import FindingCode


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CodeResult(_Frozen):
    code: FindingCode
    n_pos: int = Field(ge=0)
    n_neg: int = Field(ge=0)
    sens: float | None = None
    spec: float | None = None
    sens_ci: tuple[float, float] | None = None
    spec_ci: tuple[float, float] | None = None
    ece: float | None = None
    aurc: float | None = None


class SplitResult(_Frozen):
    model_id: str
    split: str
    n_images: int = Field(ge=0)
    json_valid_rate: float = Field(ge=0, le=1)
    codes: tuple[CodeResult, ...]

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "SplitResult":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
