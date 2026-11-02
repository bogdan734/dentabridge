from pathlib import Path

import pytest
from PIL import Image

from dentabridge_core import FindingCode
from dentabridge_ml.data.mendeley import load_mendeley


def _write(root: Path, split: str, stem: str, label_lines: list[str] | None) -> None:
    base = root / "mendeley_gingivitis" / "Dataset" / split
    (base / "Images").mkdir(parents=True, exist_ok=True)
    (base / "Labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 40), (190, 120, 120)).save(base / "Images" / f"{stem}.jpg")
    if label_lines is not None:
        (base / "Labels" / f"{stem}.txt").write_text("\n".join(label_lines) + "\n")


def _captions(root: Path, split: str, rows: list[tuple[str, int, str]]) -> None:
    lines = ["FileName,#,BS1", *(f'{name},{n},"{text}"' for name, n, text in rows)]
    path = root / "mendeley_gingivitis" / "Dataset" / split / f"{split}_captions.csv"
    path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8-sig"
    )  # the publisher's files have a BOM


HEALTHY, LEVEL_1, LEVEL_3 = (
    "2 0.30 0.40 0.10 0.20",
    "3 0.50 0.50 0.20 0.20",
    "5 0.80 0.60 0.10 0.20",
)
LEVEL_4 = "6 0.10 0.20 0.10 0.10"
JAWS = ["0 0.50 0.30 0.90 0.30", "1 0.50 0.75 0.90 0.30"]  # one box per jaw in every real file


def test_loader_reads_levels_healthy_teeth_and_captions(tmp_path: Path) -> None:
    _write(tmp_path, "Test", "00929", [*JAWS, HEALTHY, LEVEL_1, LEVEL_3, LEVEL_4])
    _write(tmp_path, "Test", "00930", [*JAWS, HEALTHY, HEALTHY])
    _captions(tmp_path, "Test", [("00929.jpg", 1, "first"), ("00929.jpg", 2, "second")])

    records = {r.record_id: r for r in load_mendeley(tmp_path)}
    sick, well = (
        records["mendeley_gingivitis/Test/00929"],
        records["mendeley_gingivitis/Test/00930"],
    )
    assert sick.codes_present() == {FindingCode.GINGIVITIS}
    assert {b.subtype for b in sick.boxes} == {"level_1", "level_3", "level_4"}
    assert len(sick.boxes) == 3, "healthy teeth and the two jaw boxes get no region"
    box = next(b for b in sick.boxes if b.subtype == "level_1")
    assert (box.x0, box.y0, box.x1, box.y1) == pytest.approx((0.4, 0.4, 0.6, 0.6))
    assert sick.path == "Dataset/Test/Images/00929.jpg"
    assert (sick.width, sick.height) == (80, 40)
    assert sick.caption == "first"
    assert well.codes_present() == frozenset(), "all teeth labelled healthy: a real negative"
    assert well.caption is None


def test_an_image_without_labelled_teeth_is_not_a_negative(tmp_path: Path) -> None:
    _write(tmp_path, "Test", "00001", [])
    _write(tmp_path, "Test", "00002", None)
    _write(tmp_path, "Test", "00003", JAWS)  # the jaws are outlined, no tooth is judged
    assert load_mendeley(tmp_path) == []


def test_consecutive_numbers_form_one_capture_sequence_across_publisher_splits(
    tmp_path: Path,
) -> None:
    for split, stem in (
        ("Training", "00010"),
        ("Training", "00011"),
        ("Test", "00012"),
        ("Test", "00020"),
    ):
        _write(tmp_path, split, stem, [HEALTHY])
    sequence = {r.path.split("/")[-1]: r.capture_sequence for r in load_mendeley(tmp_path)}
    assert sequence["00010.jpg"] == sequence["00011.jpg"] == sequence["00012.jpg"]
    assert sequence["00020.jpg"] != sequence["00010.jpg"]
    assert all(s is not None for s in sequence.values())


def test_an_unknown_class_is_an_error_not_a_guess(tmp_path: Path) -> None:
    _write(tmp_path, "Test", "00001", ["9 0.5 0.5 0.1 0.1"])
    with pytest.raises(ValueError, match="class id 9"):
        load_mendeley(tmp_path)
