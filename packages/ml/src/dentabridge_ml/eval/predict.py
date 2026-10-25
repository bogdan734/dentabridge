"""Run a VisionBackend over one split and keep every answer, valid or not.

The test split can be read only with final=True, and each such run is appended to the access
log. That log, plus the release gate being committed first, is the evidence that the test numbers
were looked at once.
"""

import hashlib
import io
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict

from dentabridge_core import SanitisedImage, VisionBackend, VisionOutput
from dentabridge_ml.data.records import ImageRecord
from dentabridge_ml.data.sources import SOURCES
from dentabridge_ml.serving.parse import ParseError, ReleaseConfig
from dentabridge_ml.splits.manifest import SplitEntry

MAX_SIDE = 1536


class PredictionRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    output: VisionOutput | None = None
    error: str | None = None  # a ParseError: the model did not answer in the agreed format


def research_release() -> ReleaseConfig:
    """Research runs let the model judge every code a published source labels; the release gate
    later decides which of them may ship."""
    covered = frozenset().union(*(s.coverage for s in SOURCES.values() if s.published))
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


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


async def predict(
    backend: VisionBackend,
    records: Sequence[ImageRecord],
    entries: Sequence[SplitEntry],
    root: Path,
    out: Path,
    final: bool = False,
    access_log: Path | None = None,
) -> Path:
    if any(e.split == "test" for e in entries):
        if not final:
            raise PermissionError("the test split is read once, with final=True")
        if access_log is None:
            raise PermissionError("a final run must name the test access log")
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        line = f"{stamp} {backend.model.model_id} commit={_commit()} split=test\n"
        with access_log.open("a", encoding="utf-8") as log:
            log.write(line)
    wanted = {e.record_id for e in entries}
    rows: list[PredictionRow] = []
    for record in records:
        if record.record_id not in wanted:
            continue
        image = sanitise(root / record.source / record.path)
        try:
            output = await backend.analyse(image, "intraoral_photo")
            rows.append(PredictionRow(record_id=record.record_id, output=output))
        except ParseError as error:
            rows.append(PredictionRow(record_id=record.record_id, error=str(error)))
    out.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    return out


def read_predictions(path: Path) -> list[PredictionRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PredictionRow.model_validate_json(line) for line in lines if line.strip()]
