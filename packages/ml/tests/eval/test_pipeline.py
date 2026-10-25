import asyncio
from pathlib import Path

import pytest
from PIL import Image

from dentabridge_core import Finding, FindingCode, ModelInfo, Region, VisionOutput
from dentabridge_ml.data.records import ImageRecord, LabelBox
from dentabridge_ml.eval.predict import predict, read_predictions, sanitise
from dentabridge_ml.eval.report import build_result
from dentabridge_ml.serving.parse import ParseError
from dentabridge_ml.splits.manifest import SplitEntry, SplitName

CARIES = FindingCode.CARIES


class ScriptedBackend:
    """Answers by image hash; None means the model produced something unparseable."""

    model = ModelInfo(model_id="scripted", version="0", backend="fixture")

    def __init__(self, answers: dict[str, VisionOutput | None]) -> None:
        self._answers = answers

    async def analyse(self, image, modality):  # type: ignore[no-untyped-def]
        answer = self._answers[image.sha256]
        if answer is None:
            raise ParseError("not the agreed JSON")
        return answer


def _output(possible: bool) -> VisionOutput:
    finding = (
        Finding(
            code=CARIES,
            status="possible",
            score=0.9,
            regions=(Region(x0=0.1, y0=0.1, x1=0.4, y1=0.4),),
        )
        if possible
        else Finding(code=CARIES, status="not_seen", score=0.1)
    )
    return VisionOutput(image_quality="ok", findings=(finding,), latency_ms=1.0)


def _world(
    tmp_path: Path, split: SplitName = "val"
) -> tuple[list[ImageRecord], tuple[SplitEntry, ...]]:
    folder = tmp_path / "aku_caries"
    folder.mkdir()
    records = []
    for i in range(8):
        Image.new("RGB", (64, 48), (i * 30, 20, 20)).save(folder / f"{i}.jpg")
        positive = i < 4
        records.append(
            ImageRecord(
                record_id=f"aku_caries/{i}",
                source="aku_caries",
                path=f"{i}.jpg",
                width=64,
                height=48,
                boxes=(LabelBox(code=CARIES, x0=0.1, y0=0.1, x1=0.4, y1=0.4),) if positive else (),
                image_labels=frozenset({CARIES}) if positive else frozenset(),
            )
        )
    entries = tuple(
        SplitEntry(
            record_id=r.record_id,
            source=r.source,
            split=split,
            group=f"g{i}",
            group_method="cluster",
        )
        for i, r in enumerate(records)
    )
    return records, entries


def _backend(tmp_path: Path, records: list[ImageRecord]) -> ScriptedBackend:
    says_possible = {0, 1, 2, 5}  # 3 hits, 1 miss (3), 1 false alarm (5)
    answers: dict[str, VisionOutput | None] = {}
    for i, record in enumerate(records):
        sha = sanitise(tmp_path / "aku_caries" / record.path).sha256
        answers[sha] = None if i == 7 else _output(i in says_possible)
    return ScriptedBackend(answers)


def test_report_matches_a_hand_count_and_scores_only_labelled_codes(tmp_path: Path) -> None:
    records, entries = _world(tmp_path)
    out = tmp_path / "val.jsonl"
    asyncio.run(predict(_backend(tmp_path, records), records, entries, tmp_path, out))
    rows = read_predictions(out)
    assert len(rows) == 8
    result = build_result("scripted", "val", {r.record_id: r for r in records}, entries, rows)
    caries = next(c for c in result.codes if c.code == CARIES)
    assert (caries.n_pos, caries.n_neg) == (4, 4)
    assert (caries.sens, caries.spec) == (0.75, 0.75)
    assert result.json_valid_rate == 7 / 8
    assert caries.ece is None, "one unparseable answer has no score, so no calibration claim"
    filling = next(c for c in result.codes if c.code == FindingCode.FILLING)
    assert (filling.n_pos, filling.n_neg, filling.sens, filling.spec) == (0, 0, None, None)


def test_uncovered_source_codes_not_scored(tmp_path: Path) -> None:
    records, entries = _world(tmp_path)
    out = tmp_path / "val.jsonl"
    asyncio.run(predict(_backend(tmp_path, records), records, entries, tmp_path, out))
    result = build_result(
        "m", "val", {r.record_id: r for r in records}, entries, read_predictions(out)
    )
    scored = {c.code for c in result.codes if c.n_pos + c.n_neg}
    assert scored == {CARIES}


def test_test_split_requires_final_and_logs(tmp_path: Path) -> None:
    records, entries = _world(tmp_path, split="test")
    backend = _backend(tmp_path, records)
    out, log = tmp_path / "test.jsonl", tmp_path / "test-access.log"
    with pytest.raises(PermissionError, match="final=True"):
        asyncio.run(predict(backend, records, entries, tmp_path, out, access_log=log))
    with pytest.raises(PermissionError, match="access log"):
        asyncio.run(predict(backend, records, entries, tmp_path, out, final=True))
    assert not out.exists()
    asyncio.run(predict(backend, records, entries, tmp_path, out, final=True, access_log=log))
    asyncio.run(predict(backend, records, entries, tmp_path, out, final=True, access_log=log))
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert "scripted" in lines[0]
    assert "split=test" in lines[0]


def test_val_never_touches_the_log(tmp_path: Path) -> None:
    records, entries = _world(tmp_path)
    log = tmp_path / "log"
    asyncio.run(
        predict(
            _backend(tmp_path, records),
            records,
            entries,
            tmp_path,
            tmp_path / "v.jsonl",
            access_log=log,
        )
    )
    assert not log.exists()


def test_sanitise_is_upright_bounded_and_metadata_free(tmp_path: Path) -> None:
    path = tmp_path / "big.jpg"
    image = Image.new("RGB", (3200, 1600), (200, 100, 50))
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees when displayed
    image.save(path, exif=exif)
    clean = sanitise(path)
    assert max(clean.width, clean.height) == 1536
    assert (clean.width, clean.height) == (768, 1536), "EXIF rotation applied, then bounded"
    from io import BytesIO

    assert not Image.open(BytesIO(clean.data)).getexif()
    assert clean == sanitise(path)


class BatchBackend:
    """Answers in batches, remembering how big each batch was."""

    model = ModelInfo(model_id="batched", version="0", backend="fixture")

    def __init__(self, answers: dict[str, VisionOutput | None]) -> None:
        self._answers = answers
        self.batches: list[int] = []

    async def analyse(self, image, modality):  # type: ignore[no-untyped-def]
        raise AssertionError("a batch backend is called in batches")

    async def analyse_many(self, images, modality):  # type: ignore[no-untyped-def]
        self.batches.append(len(images))
        return [self._answers[i.sha256] or ParseError("not the agreed JSON") for i in images]


def test_batches_give_the_same_rows_as_one_at_a_time(tmp_path: Path) -> None:
    records, entries = _world(tmp_path)
    one_by_one = _backend(tmp_path, records)
    batched = BatchBackend(one_by_one._answers)
    asyncio.run(predict(one_by_one, records, entries, tmp_path, tmp_path / "a.jsonl"))
    asyncio.run(predict(batched, records, entries, tmp_path, tmp_path / "b.jsonl", batch_size=3))
    assert batched.batches == [3, 3, 2]
    assert (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()
    assert read_predictions(tmp_path / "b.jsonl")[7].error is not None


def test_a_batch_size_needs_a_backend_that_batches(tmp_path: Path) -> None:
    records, entries = _world(tmp_path)
    with pytest.raises(TypeError, match="analyse_many"):
        asyncio.run(
            predict(
                _backend(tmp_path, records),
                records,
                entries,
                tmp_path,
                tmp_path / "x",
                batch_size=4,
            )
        )
