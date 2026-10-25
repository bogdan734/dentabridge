"""AlphaDent (Zenodo 16582489): DSLR photos with instance-segmentation polygons.

Layout and meaning, checked against the unpacked files and the paper (Sosnin et al., 2025):
- `images/{train,valid}/p<patient>_<sex>_<age>_<n>.jpg` with a YOLO-segmentation label file of
  the same name. The publisher's split is already by patient (no patient is in both).
  `images/test/test_NNN.jpg` has no labels (hidden test of a competition) and is not used.
- Classes: 0 abrasion, 1 filling, 2 crown, 3-8 caries "class 1-6". Those six are caries graded
  by SITE (Black's classes plus the WHO sixth), not by severity, so the subtype is kept as
  `caries_N` and nothing treats a higher N as worse.
- Every labelled image has a label file; an empty one means none of the nine classes was found.
"""

import re
from pathlib import Path

from dentabridge_core import FindingCode
from dentabridge_ml.data.records import (
    ImageRecord,
    LabelBox,
    image_size,
    non_empty_lines,
    parse_yolo_polygon,
)
from dentabridge_ml.data.sources import SourceId

SOURCE: SourceId = "alphadent"
SPLITS = ("train", "valid")
_NAME = re.compile(r"^(p\d+)_[FM]_\d+_\d+$")
_CODE = {0: FindingCode.ABRASION, 1: FindingCode.FILLING, 2: FindingCode.CROWN}
_FIRST_CARIES, _LAST_CARIES = 3, 8


def _code_and_subtype(cls: int) -> tuple[FindingCode, str | None]:
    if cls in _CODE:
        return _CODE[cls], None
    if _FIRST_CARIES <= cls <= _LAST_CARIES:
        return FindingCode.CARIES, f"caries_{cls - _FIRST_CARIES + 1}"
    raise ValueError(f"unknown AlphaDent class id {cls}")


def _parse(line: str) -> tuple[FindingCode, LabelBox | None]:
    """The code a line asserts, and its region unless the polygon has no area (a single point
    occurs in the real files: the class was named but nothing was outlined)."""
    code, subtype = _code_and_subtype(int(line.split()[0]))
    try:
        _, x0, y0, x1, y1 = parse_yolo_polygon(line)
    except ValueError:
        return code, None
    if not (x0 < x1 and y0 < y1):
        return code, None
    return code, LabelBox(code=code, x0=x0, y0=y0, x1=x1, y1=y1, subtype=subtype)


def load_alphadent(root: Path) -> list[ImageRecord]:
    base = root / SOURCE
    records: list[ImageRecord] = []
    for split in SPLITS:
        for image in sorted((base / "images" / split).glob("*.jpg")):
            named = _NAME.match(image.stem)
            if named is None:
                raise ValueError(f"unexpected AlphaDent file name: {image.name}")
            label = base / "labels" / split / f"{image.stem}.txt"
            parsed = [_parse(line) for line in non_empty_lines(label)]
            boxes = tuple(box for _, box in parsed if box is not None)
            width, height = image_size(image)
            records.append(
                ImageRecord(
                    record_id=f"{SOURCE}/{split}/{image.stem}",
                    source=SOURCE,
                    path=image.relative_to(base).as_posix(),
                    width=width,
                    height=height,
                    patient_id=named.group(1),
                    boxes=boxes,
                    image_labels=frozenset(code for code, _ in parsed),
                )
            )
    return records
