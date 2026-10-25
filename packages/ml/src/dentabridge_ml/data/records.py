"""One canonical record per labelled image, whatever the source's file layout."""

from collections.abc import Sequence
from pathlib import Path
from typing import Self

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dentabridge_core import FindingCode
from dentabridge_ml.data.sources import SourceId

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LabelBox(_Frozen):
    """A labelled region in image fractions, origin top left."""

    code: FindingCode
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    subtype: str | None = None  # the source's own label, e.g. "caries_3" or "D"

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("a box needs x0 < x1 and y0 < y1")
        return self


class ImageRecord(_Frozen):
    record_id: str
    source: SourceId
    path: str  # relative to <root>/<source>/
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    patient_id: str | None = None  # a real participant id from the publisher
    # Photos of one person when no real id exists (the capture order of a session).
    capture_sequence: str | None = None
    view: str | None = None  # e.g. "Frontal", "Maxillary_Occlusal": a shortcut worth reporting
    boxes: tuple[LabelBox, ...] = ()
    image_labels: frozenset[FindingCode] = frozenset()
    caption: str | None = None

    def codes_present(self) -> frozenset[FindingCode]:
        return self.image_labels | {box.code for box in self.boxes}


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:  # header only, the pixels are not decoded
        return image.size


def parse_yolo_box(line: str) -> tuple[int, float, float, float, float]:
    """`class cx cy w h` (fractions) to `class x0 y0 x1 y1`, clipped to the image."""
    cls, cx, cy, w, h = line.split()
    x, y, bw, bh = float(cx), float(cy), float(w), float(h)
    return int(cls), _clip(x - bw / 2), _clip(y - bh / 2), _clip(x + bw / 2), _clip(y + bh / 2)


def parse_yolo_polygon(line: str) -> tuple[int, float, float, float, float]:
    """`class x1 y1 x2 y2 ...` (fractions) to the polygon's bounding box."""
    cls, *numbers = line.split()
    if len(numbers) < 6 or len(numbers) % 2:
        raise ValueError(f"not a polygon: {line!r}")
    values = [float(v) for v in numbers]
    xs, ys = values[0::2], values[1::2]
    return int(cls), max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys))


def non_empty_lines(path: Path) -> Sequence[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
