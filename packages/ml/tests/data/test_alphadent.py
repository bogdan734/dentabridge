from pathlib import Path

import pytest
from PIL import Image

from dentabridge_core import FindingCode
from dentabridge_ml.data.alphadent import load_alphadent


def _write(root: Path, split: str, stem: str, label_lines: list[str]) -> None:
    base = root / "alphadent"
    (base / "images" / split).mkdir(parents=True, exist_ok=True)
    (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 40), (150, 120, 110)).save(base / "images" / split / f"{stem}.jpg")
    (base / "labels" / split / f"{stem}.txt").write_text("\n".join(label_lines) + "\n")


SQUARE = "0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6"


def test_loader_golden_per_format(tmp_path: Path) -> None:
    _write(tmp_path, "train", "p002_F_52_001", [f"7 {SQUARE}", f"1 {SQUARE}", f"0 {SQUARE}"])
    _write(tmp_path, "train", "p002_F_52_002", [])
    _write(tmp_path, "valid", "p011_M_8_001", [f"2 {SQUARE}", f"3 {SQUARE}"])
    (tmp_path / "alphadent" / "images" / "test").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "alphadent" / "images" / "test" / "test_000.jpg")

    records = {r.record_id: r for r in load_alphadent(tmp_path)}
    assert set(records) == {
        "alphadent/train/p002_F_52_001",
        "alphadent/train/p002_F_52_002",
        "alphadent/valid/p011_M_8_001",
    }, "the unlabelled test images are not loaded"
    first = records["alphadent/train/p002_F_52_001"]
    assert first.patient_id == "p002"
    assert (first.width, first.height) == (80, 40)
    by_code = {b.code: b for b in first.boxes}
    assert set(by_code) == {FindingCode.CARIES, FindingCode.FILLING, FindingCode.ABRASION}
    assert by_code[FindingCode.CARIES].subtype == "caries_5"
    caries = by_code[FindingCode.CARIES]
    assert (caries.x0, caries.y0, caries.x1, caries.y1) == (0.1, 0.2, 0.3, 0.6)
    assert records["alphadent/train/p002_F_52_002"].codes_present() == frozenset()
    assert records["alphadent/valid/p011_M_8_001"].codes_present() == {
        FindingCode.CROWN,
        FindingCode.CARIES,
    }
    assert {b.subtype for b in records["alphadent/valid/p011_M_8_001"].boxes} == {None, "caries_1"}


def test_a_polygon_without_an_area_keeps_the_label_but_drops_the_region(tmp_path: Path) -> None:
    lines = ["5 0.65442251 0.75794956", "3 0.5 0.5 0.5 0.5 0.5 0.5", f"1 {SQUARE}"]
    _write(tmp_path, "train", "p001_F_30_001", lines)
    (record,) = load_alphadent(tmp_path)
    assert [b.code for b in record.boxes] == [FindingCode.FILLING]
    assert FindingCode.CARIES in record.image_labels, "asserted by a line without an area"
    assert record.codes_present() == {FindingCode.CARIES, FindingCode.FILLING}


def test_unknown_class_or_name_fails_loudly(tmp_path: Path) -> None:
    _write(tmp_path, "train", "p001_F_30_001", [f"9 {SQUARE}"])
    with pytest.raises(ValueError, match="unknown AlphaDent class"):
        load_alphadent(tmp_path)
    other = tmp_path / "other"
    _write(other, "train", "stranger", [])
    with pytest.raises(ValueError, match="unexpected AlphaDent file name"):
        load_alphadent(other)
