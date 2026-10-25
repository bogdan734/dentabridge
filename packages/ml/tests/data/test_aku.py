import json
from pathlib import Path

from PIL import Image

from dentabridge_core import FindingCode
from dentabridge_ml.data.aku import load_aku

W, H = 200, 100


def _image(root: Path, protocol: str, view: str, name: str) -> str:
    folder = root / "aku_caries" / "Dataset" / "Images" / protocol / view
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (W, H), (120, 80, 70)).save(folder / f"{name}.jpg")
    return name


def _label(
    root: Path, protocol: str, view: str, name: str, *shapes: dict[str, object], **extra: int
) -> None:
    folder = root / "aku_caries" / "Dataset" / "Annotations" / "Labelme" / protocol / view
    folder.mkdir(parents=True, exist_ok=True)
    document = {"imageWidth": W, "imageHeight": H, "shapes": list(shapes)} | extra
    (folder / f"{name}.json").write_text(json.dumps(document))


def _rect(label: str, a: tuple[float, float], b: tuple[float, float]) -> dict[str, object]:
    return {"label": label, "points": [list(a), list(b)], "shape_type": "rectangle"}


def _by_name(records: list) -> dict:  # type: ignore[type-arg]
    return {r.record_id.rsplit("/", 1)[-1]: r for r in records}


def test_loader_golden_per_format(tmp_path: Path) -> None:
    root = tmp_path
    hyphen = _image(
        root, "retractors", "Occlusal", "anonymous_003-007-1002-01_1731041206649_Occlusal_View"
    )
    underscore = _image(
        root, "no_retractors", "Frontal", "anonymous_003_008_519_00_1728731108258_Frontal_View"
    )
    healthy = _image(
        root, "retractors", "Frontal", "anonymous_003-007-1002-01_1731041206649_Frontal_View"
    )
    _label(
        root,
        "retractors",
        "Occlusal",
        hyphen,
        _rect("D", (20, 10), (60, 50)),
        _rect("d", (150, 90), (100, 20)),
    )
    _label(root, "no_retractors", "Frontal", underscore, _rect("M", (1, 1), (5, 5)))

    records = _by_name(load_aku(root))
    positive = records[hyphen]
    assert positive.patient_id == "003-007-1002"
    assert positive.view == "Occlusal"
    assert positive.image_labels == {FindingCode.CARIES}
    first, second = positive.boxes
    assert (first.x0, first.y0, first.x1, first.y1, first.subtype) == (0.1, 0.1, 0.3, 0.5, "D")
    assert (second.x0, second.y0, second.x1, second.y1, second.subtype) == (
        0.5,
        0.2,
        0.75,
        0.9,
        "d",
    )
    assert records[underscore].patient_id == "003-008-519"
    assert records[underscore].image_labels == frozenset()  # only the unexplained M class
    assert records[underscore].boxes == ()
    assert records[healthy].patient_id == "003-007-1002"  # same person, other view
    assert records[healthy].image_labels == frozenset()
    assert records[healthy].boxes == ()


def test_shapes_outside_the_image_keep_the_label_but_drop_the_box(tmp_path: Path) -> None:
    name = _image(
        tmp_path, "retractors", "Frontal", "anonymous_003-007-1141-01_1732595713489_Frontal_View"
    )
    _label(
        tmp_path,
        "retractors",
        "Frontal",
        name,
        _rect("D", (10, 10), (50, 40)),
        _rect("d", (20, 300), (40, 320)),
    )
    (record,) = load_aku(tmp_path)
    assert record.image_labels == {FindingCode.CARIES}
    assert len(record.boxes) == 1
    assert record.boxes[0].subtype == "D"


def test_a_label_file_that_disagrees_about_the_image_size_gives_no_boxes(tmp_path: Path) -> None:
    name = _image(tmp_path, "retractors", "Frontal", "anonymous_003-007-7-01_1_Frontal_View")
    folder = (
        tmp_path / "aku_caries" / "Dataset" / "Annotations" / "Labelme" / "retractors" / "Frontal"
    )
    folder.mkdir(parents=True)
    wrong = {"imageWidth": H, "imageHeight": W, "shapes": [_rect("D", (10, 10), (50, 40))]}
    (folder / f"{name}.json").write_text(json.dumps(wrong))
    (record,) = load_aku(tmp_path)
    assert record.image_labels == {FindingCode.CARIES}
    assert record.boxes == ()


def test_pilot_images_are_grouped_by_capture_order(tmp_path: Path) -> None:
    views = ["frontal", "rightLateral", "leftLateral", "maxillary", "mandibular"]
    base = 1_726_650_174_308
    names: list[str] = []
    for start in [base, base + 3_600_000, base + 7_200_000]:
        for step, view in enumerate(views):
            names.append(
                _image(
                    tmp_path,
                    "pilot",
                    view.capitalize(),
                    f"anonymous-{view}View-{start + step * 90_000}",
                )
            )
    records = _by_name(load_aku(tmp_path))
    sequences = [records[n].capture_sequence for n in names]
    assert sequences == ["pilot-0000"] * 5 + ["pilot-0001"] * 5 + ["pilot-0002"] * 5
    assert all(records[n].patient_id is None for n in names)
