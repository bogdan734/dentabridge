from pathlib import Path

import pytest

from dentabridge_core import FindingCode
from dentabridge_ml.data.records import ImageRecord, LabelBox
from dentabridge_ml.splits.manifest import SplitEntry, SplitManifest, write_manifests
from dentabridge_ml.train import __main__ as cli
from dentabridge_ml.train.data import read_sft


def _records() -> list[ImageRecord]:
    return [
        ImageRecord(
            record_id=f"alphadent/r{i}",
            source="alphadent",
            path=f"r{i}.jpg",
            width=100,
            height=80,
            boxes=(LabelBox(code=FindingCode.CARIES, x0=0.1, y0=0.1, x1=0.4, y1=0.4),),
        )
        for i in range(10)
    ]


@pytest.fixture
def manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    records = _records()
    monkeypatch.setattr(cli, "load_records", lambda source, root: records)
    entries = tuple(
        SplitEntry(
            record_id=r.record_id,
            source="alphadent",
            split="train" if i < 6 else "val" if i < 8 else "test",
            group=f"p:{i}",
            group_method="patient_id",
        )
        for i, r in enumerate(records)
    )
    write_manifests(SplitManifest(seed=0, ratios=(0.6, 0.2, 0.2), entries=entries), tmp_path / "m")
    return tmp_path / "m"


def _sft(manifests: Path, out: Path, *extra: str) -> int:
    return cli.main(["sft", "train", "--manifests", str(manifests), "--out", str(out), *extra])


def test_sft_writes_the_chosen_split_only(manifests: Path, tmp_path: Path) -> None:
    out = tmp_path / "train.jsonl"
    assert _sft(manifests, out, "--negatives", "0") == 0
    assert {r.id for r in read_sft(out)} == {f"alphadent/r{i}" for i in range(6)}


def test_sft_limit_keeps_a_pilot_small(manifests: Path, tmp_path: Path) -> None:
    out = tmp_path / "pilot.jsonl"
    _sft(manifests, out, "--limit", "4")
    assert len(read_sft(out)) == 4


def test_sft_never_builds_training_rows_from_the_test_split(
    manifests: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        cli.main(["sft", "test", "--manifests", str(manifests), "--out", str(tmp_path / "x.jsonl")])


def test_sft_stops_on_an_edited_manifest(manifests: Path, tmp_path: Path) -> None:
    (manifests / "val.jsonl").write_text("")
    with pytest.raises(ValueError, match="does not match"):
        _sft(manifests, tmp_path / "x.jsonl")
