"""Mendeley gingivitis set (3253gj88rr): intraoral photos with one YOLO box per tooth.

Layout, read from the archive (the class meaning from its captions, for example "5 non-inflamed
teeth at position 13, 12, 21, 22, 23 and 1 level of gingivitis 1 at position 11"):
- `Dataset/<Split>/Images/NNNNN.jpg` with `Dataset/<Split>/Labels/NNNNN.txt` (`class cx cy w h`
  per tooth) and `Dataset/<Split>/<Split>_captions.csv` (`FileName,#,BS1`: three English captions
  per image, written by periodontists, UTF-8 with a BOM).
- Class 2 is a tooth with healthy gums, classes 3, 4, 5 are gingivitis of level 1, 2, 3. Any other
  class is refused until its meaning is checked.
- The file numbers have no patient behind them, but the publisher kept blocks of consecutive
  numbers together in one split, so a run of consecutive numbers is treated as one capture
  sequence (a patient or a visit) and never split across train and test.
"""

import csv
from pathlib import Path

from dentabridge_core import FindingCode
from dentabridge_ml.data.records import (
    ImageRecord,
    LabelBox,
    image_size,
    non_empty_lines,
    parse_yolo_box,
)
from dentabridge_ml.data.sources import SourceId

SOURCE: SourceId = "mendeley_gingivitis"
HEALTHY = 2
LEVELS = {3: "level_1", 4: "level_2", 5: "level_3"}


def _boxes(label: Path) -> tuple[bool, tuple[LabelBox, ...]]:
    """Whether any tooth is labelled at all, and the boxes of the inflamed ones."""
    lines = non_empty_lines(label)
    boxes = []
    for line in lines:
        cls, x0, y0, x1, y1 = parse_yolo_box(line)
        if cls in LEVELS:
            boxes.append(
                LabelBox(
                    code=FindingCode.GINGIVITIS, x0=x0, y0=y0, x1=x1, y1=y1, subtype=LEVELS[cls]
                )
            )
        elif cls != HEALTHY:
            raise ValueError(f"unknown Mendeley class id {cls} in {label}")
    return bool(lines), tuple(boxes)


def _captions(folder: Path) -> dict[str, str]:
    """The first caption of each image."""
    first: dict[str, str] = {}
    for path in folder.glob("*_captions.csv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                first.setdefault(row["FileName"], row["BS1"])
    return first


def _sequences(numbers: list[int]) -> dict[int, str]:
    """Name each run of consecutive numbers after its first member."""
    names: dict[int, str] = {}
    start = previous = None
    for number in sorted(set(numbers)):
        if previous is None or number != previous + 1:
            start = number
        names[number] = f"{SOURCE}:{start:05d}"
        previous = number
    return names


def load_mendeley(root: Path) -> list[ImageRecord]:
    base = root / SOURCE
    found: list[tuple[Path, Path, bool, tuple[LabelBox, ...]]] = []
    captions: dict[str, str] = {}
    for images in sorted((base / "Dataset").glob("*/Images")):
        split_dir = images.parent
        captions.update(_captions(split_dir))
        for image in sorted(images.glob("*.jpg")):
            label = split_dir / "Labels" / f"{image.stem}.txt"
            if not label.exists():
                continue
            labelled, boxes = _boxes(label)
            if labelled:  # an empty label file says nothing about the teeth: not a negative
                found.append((image, split_dir, labelled, boxes))
    sequence = _sequences([int(image.stem) for image, *_ in found])
    records = []
    for image, split_dir, _, boxes in found:
        width, height = image_size(image)
        records.append(
            ImageRecord(
                record_id=f"{SOURCE}/{split_dir.name}/{image.stem}",
                source=SOURCE,
                path=image.relative_to(base).as_posix(),
                width=width,
                height=height,
                capture_sequence=sequence[int(image.stem)],
                boxes=boxes,
                caption=captions.get(image.name),
            )
        )
    return records
