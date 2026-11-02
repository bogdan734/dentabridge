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
            record_id=f"{source}/r{i}",
            source=source,
            path=f"r{i}.jpg",
            width=100,
            height=80,
            boxes=(LabelBox(code=FindingCode.CARIES, x0=0.1, y0=0.1, x1=0.4, y1=0.4),),
        )
        for i in range(10)
        for source in ("alphadent", "aku_caries")
    ]


@pytest.fixture
def manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    records = _records()
    monkeypatch.setattr(
        cli, "load_records", lambda source, root: [r for r in records if r.source == source]
    )
    entries = tuple(
        SplitEntry(
            record_id=r.record_id,
            source=r.source,
            split="train" if i < 12 else "val" if i < 16 else "test",
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
    ids = {f"{s}/r{i}" for i in range(6) for s in ("alphadent", "aku_caries")}
    assert {r.id for r in read_sft(out)} == ids


def test_sft_can_be_limited_to_some_sources(manifests: Path, tmp_path: Path) -> None:
    out = tmp_path / "aku.jsonl"
    _sft(manifests, out, "--negatives", "0", "--sources", "aku_caries")
    rows = read_sft(out)
    assert len(rows) == 6
    assert {r.id.split("/")[0] for r in rows} == {"aku_caries"}


def test_sft_repeat_multiplies_one_source(manifests: Path, tmp_path: Path) -> None:
    out = tmp_path / "repeated.jsonl"
    _sft(manifests, out, "--negatives", "0", "--repeat", "alphadent=3")
    rows = read_sft(out)
    assert sum(r.id.startswith("alphadent/") for r in rows) == 18
    assert sum(r.id.startswith("aku_caries/") for r in rows) == 6
    with pytest.raises(SystemExit):
        _sft(manifests, out, "--repeat", "alphadent")
    with pytest.raises(SystemExit):
        _sft(manifests, out, "--repeat", "alphadent=0")


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
