from dentabridge_core import Finding, FindingCode, VisionOutput
from dentabridge_ml.data.records import ImageRecord, LabelBox
from dentabridge_ml.eval.predict import PredictionRow
from dentabridge_ml.eval.slices import slice_results
from dentabridge_ml.splits.manifest import SplitEntry

CARIES = FindingCode.CARIES


def _record(i: int, source: str, view: str | None, positive: bool) -> ImageRecord:
    box = LabelBox(code=CARIES, x0=0.1, y0=0.1, x1=0.4, y1=0.4)
    return ImageRecord(
        record_id=f"{source}/{i}",
        source=source,  # type: ignore[arg-type]
        path=f"{i}.jpg",
        width=10,
        height=10,
        view=view,
        boxes=(box,) if positive else (),
    )


def _row(record: ImageRecord, says_possible: bool) -> PredictionRow:
    status = "possible" if says_possible else "not_seen"
    finding = Finding(code=CARIES, status=status, score=0.9 if says_possible else 0.1)  # type: ignore[arg-type]
    return PredictionRow(
        record_id=record.record_id,
        output=VisionOutput(image_quality="ok", findings=(finding,), latency_ms=1.0),
    )


def _world() -> tuple[dict[str, ImageRecord], list[SplitEntry], list[PredictionRow]]:
    records = [
        _record(0, "aku_caries", "Frontal", True),
        _record(1, "aku_caries", "Frontal", False),
        _record(2, "aku_caries", "Mandibular", True),
        _record(3, "aku_caries", "Mandibular", False),
        _record(4, "alphadent", None, True),
        _record(5, "alphadent", None, False),
    ]
    says = [True, False, False, False, True, True]
    entries = [
        SplitEntry(
            record_id=r.record_id,
            source=r.source,
            split="val",
            group=f"g{i}",
            group_method="cluster",
        )
        for i, r in enumerate(records)
    ]
    return (
        {r.record_id: r for r in records},
        entries,
        [_row(r, s) for r, s in zip(records, says, strict=True)],
    )


def test_slices_by_source_score_each_source_on_its_own_images() -> None:
    records, entries, rows = _world()
    sliced = slice_results("m", "val", records, entries, rows, by="source")
    assert set(sliced) == {"aku_caries", "alphadent"}
    aku = next(c for c in sliced["aku_caries"].codes if c.code == CARIES)
    alpha = next(c for c in sliced["alphadent"].codes if c.code == CARIES)
    assert (aku.n_pos, aku.n_neg, aku.sens, aku.spec) == (2, 2, 0.5, 1.0)
    assert (alpha.n_pos, alpha.n_neg, alpha.sens, alpha.spec) == (1, 1, 1.0, 0.0)
    assert sliced["aku_caries"].n_images == 4


def test_slices_by_view_name_missing_views_unknown() -> None:
    records, entries, rows = _world()
    sliced = slice_results("m", "val", records, entries, rows, by="view")
    assert set(sliced) == {"Frontal", "Mandibular", "unknown"}
    assert sliced["unknown"].n_images == 2
    assert sliced["Frontal"].split == "val [view=Frontal]"
