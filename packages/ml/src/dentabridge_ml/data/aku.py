"""AKU caries set (Zenodo 14827784): five views per participant, LabelMe rectangles for decay.

What the audit of the unpacked files found, and what this loader does about it:
- Names carry a participant id in two spellings (`003-007-1002-01` and `003_008_519_00`), one
  protocol per participant, one timestamp for all five views. The 760 "pilot" images have no id,
  only a capture time, taken view by view (frontal first).
- An image with a LabelMe file holds decay (`D` permanent, `d` primary tooth). An image without
  one is treated as free of decay, the convention of the publisher's own detection benchmark.
- 42 of 6,827 shapes lie outside their image (annotated in another, rotated frame). The image
  still counts as positive; the box is dropped. `M`/`m` (45 shapes) are not explained by the
  publisher and are ignored.
"""

import json
import re
from pathlib import Path

from dentabridge_core import FindingCode
from dentabridge_ml.data.records import ImageRecord, LabelBox, image_size
from dentabridge_ml.data.sources import SourceId

SOURCE: SourceId = "aku_caries"
DECAY = {"D", "d"}
TOLERANCE_PX = 1.0
PILOT_GAP_MS = 15 * 60 * 1000  # a longer pause between two captures starts a new participant

_NAMED = re.compile(r"^anonymous_(\d{3})[-_](\d{3})[-_](\d+)[-_]\d{2}_\d+_\w+_View$")
_PILOT = re.compile(r"^anonymous-(\w+)View-(\d+)")


def _pilot_sequences(images: list[Path]) -> dict[Path, str]:
    captures = sorted(
        (int(m.group(2)), m.group(1), path)
        for path in images
        if (m := _PILOT.match(path.stem)) is not None
    )
    sequences: dict[Path, str] = {}
    number, last = -1, None
    for stamp, view, path in captures:
        if last is None or view == "frontal" or stamp - last > PILOT_GAP_MS:
            number += 1
        sequences[path] = f"pilot-{number:04d}"
        last = stamp
    return sequences


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _read_labelme(
    path: Path, width: int, height: int
) -> tuple[tuple[LabelBox, ...], frozenset[FindingCode]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    declared_size_matches = (data["imageWidth"], data["imageHeight"]) == (width, height)
    boxes: list[LabelBox] = []
    positive = False
    for shape in data["shapes"]:
        if shape["label"] not in DECAY:
            continue
        positive = True
        if shape["shape_type"] != "rectangle" or not declared_size_matches:
            continue
        (xa, ya), (xb, yb) = shape["points"][:2]
        x0, x1, y0, y1 = min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)
        if x0 < -TOLERANCE_PX or y0 < -TOLERANCE_PX:
            continue
        if x1 > width + TOLERANCE_PX or y1 > height + TOLERANCE_PX:
            continue
        box = (_clamp(x0 / width), _clamp(y0 / height), _clamp(x1 / width), _clamp(y1 / height))
        if box[0] < box[2] and box[1] < box[3]:
            boxes.append(
                LabelBox(
                    code=FindingCode.CARIES,
                    x0=box[0],
                    y0=box[1],
                    x1=box[2],
                    y1=box[3],
                    subtype=shape["label"],
                )
            )
    return tuple(boxes), frozenset({FindingCode.CARIES}) if positive else frozenset()


def load_aku(root: Path) -> list[ImageRecord]:
    base = root / SOURCE / "Dataset"
    images = sorted((base / "Images").rglob("*.jpg"))
    sequences = _pilot_sequences(images)
    records: list[ImageRecord] = []
    for path in images:
        protocol, view = path.parent.parent.name, path.parent.name
        width, height = image_size(path)
        named = _NAMED.match(path.stem)
        label_file = base / "Annotations" / "Labelme" / protocol / view / f"{path.stem}.json"
        boxes, labels = (
            _read_labelme(label_file, width, height) if label_file.exists() else ((), frozenset())
        )
        records.append(
            ImageRecord(
                record_id=f"{SOURCE}/{protocol}/{view}/{path.stem}",
                source=SOURCE,
                path=path.relative_to(root / SOURCE).as_posix(),
                width=width,
                height=height,
                patient_id="-".join(named.groups()) if named else None,
                capture_sequence=sequences.get(path),
                view=view,
                boxes=boxes,
                image_labels=labels,
            )
        )
    return records
