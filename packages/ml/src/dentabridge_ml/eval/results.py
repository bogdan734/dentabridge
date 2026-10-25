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


def _value(value: float | None, interval: tuple[float, float] | None = None) -> str:
    if value is None:
        return "—"
    if interval is None:
        return f"{value:.2f}"
    return f"{value:.2f} ({interval[0]:.2f}-{interval[1]:.2f})"


def render_markdown(result: SplitResult) -> str:
    """The README and model-card table: 95 % cluster-bootstrap intervals, dashes where a number
    cannot be given (no positives, no negatives, no score)."""
    valid = f"{result.json_valid_rate * 100:.1f} %"
    lines = [
        f"# {result.model_id} on {result.split}",
        "",
        f"{result.n_images} images, {valid} of answers were valid JSON.",
        "",
        "| finding | positives | negatives | sensitivity | specificity | ECE | AURC |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in result.codes:
        cells = [
            c.code.value,
            str(c.n_pos),
            str(c.n_neg),
            _value(c.sens, c.sens_ci),
            _value(c.spec, c.spec_ci),
            _value(c.ece),
            _value(c.aurc),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
