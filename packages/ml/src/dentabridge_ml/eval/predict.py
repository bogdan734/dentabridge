"""Run a VisionBackend over one split and keep every answer, valid or not.

The test split can be read only with final=True, and each such run is appended to the access
log. That log, plus the release gate being committed first, is the evidence that the test numbers
were looked at once.
"""

import hashlib
import io
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict

from dentabridge_core import Modality, SanitisedImage, VisionBackend, VisionOutput
from dentabridge_ml.data.records import ImageRecord
from dentabridge_ml.data.sources import SOURCES, SourceId
from dentabridge_ml.provenance import git_commit
from dentabridge_ml.serving.parse import ParseError, ReleaseConfig
from dentabridge_ml.splits.manifest import SplitEntry

MAX_SIDE = 1536


class PredictionRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    output: VisionOutput | None = None
    error: str | None = None  # a ParseError: the model did not answer in the agreed format
    raw: str | None = None  # what it wrote instead, for the failure analysis


@runtime_checkable
class BatchVisionBackend(VisionBackend, Protocol):
    """A backend that answers several pictures in one forward pass (GPU evaluation)."""

    async def analyse_many(
        self, images: Sequence[SanitisedImage], modality: Modality
    ) -> list[VisionOutput | ParseError]: ...


def research_release(sources: Iterable[SourceId]) -> ReleaseConfig:
    """Research runs let the model judge every code the given sources label, no more: the prompt
    must match what training taught. The release gate later decides which codes may ship."""
    covered = frozenset().union(*(SOURCES[s].coverage for s in sources if SOURCES[s].published))
    return ReleaseConfig(assessed=covered)


def sanitise(path: Path, max_side: int = MAX_SIDE) -> SanitisedImage:
    """The same preparation the API applies to an upload: upright, RGB, bounded, no metadata."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    data = buffer.getvalue()
    return SanitisedImage(
        data=data, width=image.width, height=image.height, sha256=hashlib.sha256(data).hexdigest()
    )


async def _answer(backend: VisionBackend, image: SanitisedImage) -> VisionOutput | ParseError:
    try:
        return await backend.analyse(image, "intraoral_photo")
    except ParseError as error:
        return error


async def predict(
    backend: VisionBackend,
    records: Sequence[ImageRecord],
    entries: Sequence[SplitEntry],
    root: Path,
    out: Path,
    final: bool = False,
    access_log: Path | None = None,
    batch_size: int = 1,
) -> Path:
    if batch_size > 1 and not isinstance(backend, BatchVisionBackend):
        raise TypeError("batch_size > 1 needs a backend with analyse_many")
    if any(e.split == "test" for e in entries):
        if not final:
            raise PermissionError("the test split is read once, with final=True")
        if access_log is None:
            raise PermissionError("a final run must name the test access log")
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        line = f"{stamp} {backend.model.model_id} commit={git_commit()} split=test\n"
        with access_log.open("a", encoding="utf-8") as log:
            log.write(line)
    wanted = {e.record_id for e in entries}
    chosen = [r for r in records if r.record_id in wanted]
    rows: list[PredictionRow] = []
    for start in range(0, len(chosen), batch_size):
        chunk = chosen[start : start + batch_size]
        images = [sanitise(root / r.source / r.path) for r in chunk]
        if isinstance(backend, BatchVisionBackend) and batch_size > 1:
            results = await backend.analyse_many(images, "intraoral_photo")
        else:
            results = [await _answer(backend, image) for image in images]
        for record, result in zip(chunk, results, strict=True):
            if isinstance(result, ParseError):
                rows.append(
                    PredictionRow(record_id=record.record_id, error=str(result), raw=result.raw)
                )
            else:
                rows.append(PredictionRow(record_id=record.record_id, output=result))
    out.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    return out


def read_predictions(path: Path) -> list[PredictionRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PredictionRow.model_validate_json(line) for line in lines if line.strip()]
