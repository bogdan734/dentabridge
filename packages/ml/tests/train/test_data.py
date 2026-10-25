import json
import re
from pathlib import Path

from dentabridge_core import ASSESSABLE, FindingCode
from dentabridge_ml.data.records import ImageRecord, LabelBox
from dentabridge_ml.data.sources import SOURCES
from dentabridge_ml.splits.manifest import SplitEntry
from dentabridge_ml.targets.degrade import DEGRADE_KINDS, REASON
from dentabridge_ml.train.data import SftRow, build_rows, read_sft, write_sft

COVERED = SOURCES["alphadent"].coverage & ASSESSABLE["intraoral_photo"]


def _records(count: int) -> dict[str, ImageRecord]:
    records = {}
    for i in range(count):
        boxes = (
            [LabelBox(code=FindingCode.CARIES, x0=0.1, y0=0.1, x1=0.4, y1=0.4)]
            if i % 2 == 0
            else []
        )
        record = ImageRecord(
            record_id=f"alphadent/r{i}",
            source="alphadent",
            path=f"r{i}.jpg",
            width=100,
            height=80,
            boxes=tuple(boxes),
        )
        records[record.record_id] = record
    return records


def _train_entries(records: dict[str, ImageRecord], held_out: int = 2) -> list[SplitEntry]:
    return [
        SplitEntry(
            record_id=rid,
            source="alphadent",
            split="train",
            group=f"p:{rid}",
            group_method="patient_id",
        )
        for rid in list(records)[held_out:]
    ]


def _asked(row: SftRow) -> set[str]:
    match = re.search(r"Judge only these findings: ([a-z_, ]+)\.", row.prompt)
    assert match, row.prompt
    return set(match.group(1).split(", "))


def test_rows_cover_the_split_and_add_degraded_negatives() -> None:
    records = _records(12)
    rows = build_rows(records, _train_entries(records), negatives_frac=0.3, seed=1)
    real = [r for r in rows if r.degrade is None]
    degraded = [r for r in rows if r.degrade is not None]
    assert len(real) == 10
    assert len(degraded) == 3  # round(10 * 0.3)
    assert len({r.id for r in rows}) == len(rows)
    assert {r.id for r in real} == {
        rid for rid in records if rid not in ("alphadent/r0", "alphadent/r1")
    }
    assert all(r.id.split("#")[0] in records for r in degraded)
    assert all(r.image == f"alphadent/{records[r.id.split('#')[0]].path}" for r in rows)


def test_degraded_rows_teach_retake_with_the_matching_reason() -> None:
    records = _records(12)
    rows = build_rows(records, _train_entries(records), negatives_frac=0.4, seed=1)
    degraded = [r for r in rows if r.degrade is not None]
    assert {r.degrade.kind for r in degraded if r.degrade} == set(DEGRADE_KINDS)
    for row in degraded:
        assert row.degrade is not None
        answer = json.loads(row.answer)
        assert (answer["image_quality"], answer["quality_reason"], answer["findings"]) == (
            "unusable",
            REASON[row.degrade.kind],
            [],
        )


def test_answer_judges_exactly_the_codes_the_prompt_asks() -> None:
    records = _records(30)
    rows = build_rows(records, _train_entries(records), negatives_frac=0.0, seed=2)
    for row in rows:
        judged = {f["code"] for f in json.loads(row.answer)["findings"]}
        assert judged == _asked(row)
        assert judged <= {c.value for c in COVERED}


def test_subsets_are_optional_and_otherwise_every_row_asks_for_all_codes() -> None:
    records = _records(30)
    entries = _train_entries(records)
    full = {c.value for c in COVERED}
    flat = build_rows(records, entries, negatives_frac=0.0, seed=2, subsets=False)
    assert all(_asked(r) == full for r in flat)
    mixed = build_rows(records, entries, negatives_frac=0.0, seed=2)
    asked = [_asked(r) for r in mixed]
    assert any(a < full for a in asked), "some rows teach 'judge only these'"
    assert any(a == full for a in asked), "the full prompt used at inference is not new"


def test_rows_are_reproducible_from_the_seed() -> None:
    records = _records(20)
    entries = _train_entries(records)
    first = build_rows(records, entries, negatives_frac=0.25, seed=5)
    assert first == build_rows(records, entries, negatives_frac=0.25, seed=5)
    assert first != build_rows(records, entries, negatives_frac=0.25, seed=6)


def test_sft_file_is_one_json_row_per_line_and_reads_back(tmp_path: Path) -> None:
    records = _records(8)
    rows = build_rows(records, _train_entries(records), negatives_frac=0.5, seed=3)
    out = tmp_path / "train.jsonl"
    assert write_sft(rows, out) == len(rows)
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(rows)
    assert read_sft(out) == rows
