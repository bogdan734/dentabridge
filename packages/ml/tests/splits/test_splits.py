from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from dentabridge_core import FindingCode
from dentabridge_ml.data.records import ImageRecord, LabelBox
from dentabridge_ml.splits.group import group, group_method, hamming, phash
from dentabridge_ml.splits.manifest import SPLITS, read_entries, split, write_manifests

DIM = 16


def _record(
    i: int, *, source: str = "aku_caries", patient: str | None = None, pos: bool = False
) -> ImageRecord:
    boxes = (LabelBox(code=FindingCode.CARIES, x0=0.1, y0=0.1, x1=0.4, y1=0.4),) if pos else ()
    return ImageRecord.model_validate(
        {
            "record_id": f"{source}/img{i:04d}",
            "source": source,
            "path": f"img{i:04d}.jpg",
            "width": 10,
            "height": 10,
            "patient_id": patient,
            "boxes": boxes,
        }
    )


def _mouths(
    n_mouths: int, per_mouth: int, seed: int = 0
) -> tuple[list[ImageRecord], NDArray[np.floating], list[int]]:
    """n synthetic mouths, each photographed `per_mouth` times with small noise."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_mouths, DIM))
    records, embs, hashes = [], [], []
    for m in range(n_mouths):
        base_hash = int(rng.integers(0, 2**63))
        for k in range(per_mouth):
            records.append(_record(m * per_mouth + k, pos=m % 3 == 0))
            embs.append(centres[m] + rng.normal(scale=0.01, size=DIM))
            hashes.append(base_hash ^ (1 << int(rng.integers(0, 64))) if k else base_hash)
    return records, np.array(embs), hashes


def test_phash_is_stable_under_rescale_and_far_for_different_pictures(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
    original, other = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(pixels).save(original)
    Image.fromarray(pixels).resize((64, 48)).save(tmp_path / "small.png")
    Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(other)
    assert hamming(phash(original), phash(tmp_path / "small.png")) <= 6
    assert hamming(phash(original), phash(other)) >= 16


def test_group_joins_transitively_and_keeps_distinct_mouths_apart() -> None:
    records, emb, hashes = _mouths(40, 5)
    groups = group(records, emb, hashes, max_cos_dist=0.05, max_hamming=2)
    assert len(set(groups)) == 40
    assert all(len(set(groups[m * 5 : (m + 1) * 5])) == 1 for m in range(40))


def test_a_real_patient_id_names_the_group_a_duplicate_joins() -> None:
    records, emb, hashes = _mouths(2, 3)
    records = [
        r.model_copy(update={"patient_id": "kid-7" if i < 2 else None})
        for i, r in enumerate(records)
    ]
    groups = group(records, emb, hashes, 0.05, 2)
    # photo 2 is a near-duplicate of the two with an id, so it joins them under the real id
    assert groups[0] == groups[1] == groups[2] == "p:aku_caries:kid-7"
    assert groups[3] == groups[4] == groups[5]
    assert groups[3].startswith("c:")


def test_capture_sequence_groups_when_there_is_no_id() -> None:
    records, emb, hashes = _mouths(2, 3)
    records = [
        r.model_copy(update={"capture_sequence": f"pilot-{i // 3}"}) for i, r in enumerate(records)
    ]
    groups = group(records, emb, hashes, 0.05, 2)
    assert groups[:3] == ["s:aku_caries:pilot-0"] * 3
    assert groups[3:] == ["s:aku_caries:pilot-1"] * 3
    assert [group_method(r) for r in records] == ["capture_sequence"] * 6


def test_near_identical_photos_merge_two_people() -> None:
    a = _record(0, patient="kid-1")
    b = _record(1, patient="kid-2")
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert group([a, b], emb, [0b1010, 0b1011], 0.0, 1) == ["p:aku_caries:kid-1"] * 2
    apart = group([a, b], emb, [0, 2**40 - 1], 0.0, 1)
    assert apart == ["p:aku_caries:kid-1", "p:aku_caries:kid-2"]


def test_group_needs_aligned_inputs() -> None:
    records, emb, hashes = _mouths(2, 2)
    with pytest.raises(ValueError, match="line up"):
        group(records, emb[:-1], hashes, 0.05, 2)


def _manifest(seed: int = 5):  # type: ignore[no-untyped-def]
    records, emb, hashes = _mouths(300, 4)
    groups = group(records, emb, hashes, 0.05, 2)
    return records, groups, split(records, groups, seed)


def test_no_group_in_two_splits() -> None:
    _, groups, manifest = _manifest()
    seen: dict[str, set[str]] = {}
    for entry in manifest.entries:
        seen.setdefault(entry.group, set()).add(entry.split)
    assert all(len(s) == 1 for s in seen.values())
    assert len(set(groups)) == len(seen) == 300


def test_no_near_duplicate_across_splits() -> None:
    records, emb, hashes = _mouths(120, 4, seed=3)
    groups = group(records, emb, hashes, 0.05, 2)
    manifest = split(records, groups, seed=1)
    where = {e.record_id: e.split for e in manifest.entries}
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cosine = 1.0 - unit @ unit.T
    for i, j in zip(*np.nonzero(cosine <= 0.05), strict=True):
        assert where[records[i].record_id] == where[records[j].record_id]
    for i, a in enumerate(hashes):
        for j in range(i + 1, len(hashes)):
            if hamming(a, hashes[j]) <= 2:
                assert where[records[i].record_id] == where[records[j].record_id]


def test_manifest_deterministic_and_seed_sensitive(tmp_path: Path) -> None:
    records, groups, first = _manifest(seed=5)
    again = split(records, groups, seed=5)
    other = split(records, groups, seed=6)
    assert first.sha256() == again.sha256()
    assert first.sha256() != other.sha256()
    write_manifests(first, tmp_path / "a")
    write_manifests(again, tmp_path / "b")
    assert (tmp_path / "a" / "manifests.sha256").read_text() == (
        tmp_path / "b" / "manifests.sha256"
    ).read_text()


def test_read_entries_returns_what_was_written_and_checks_the_digest(tmp_path: Path) -> None:
    _, _, manifest = _manifest()
    write_manifests(manifest, tmp_path)
    for name in SPLITS:
        assert set(read_entries(tmp_path, name)) == set(manifest.of(name))
    (tmp_path / "val.jsonl").write_text("")  # editing one split invalidates all three
    with pytest.raises(ValueError, match=r"val\.jsonl does not match manifests\.sha256"):
        read_entries(tmp_path, "train")


def test_read_entries_needs_a_digest_for_every_split(tmp_path: Path) -> None:
    _, _, manifest = _manifest()
    write_manifests(manifest, tmp_path)
    lines = (tmp_path / "manifests.sha256").read_text().splitlines(keepends=True)
    (tmp_path / "manifests.sha256").write_text("".join(lines[:2]))
    with pytest.raises(ValueError, match=r"does not list test\.jsonl"):
        read_entries(tmp_path, "train")


def test_ratios_are_respected_per_stratum() -> None:
    records, _, manifest = _manifest()
    total = len(records)
    for name, target in zip(SPLITS, (0.8, 0.1, 0.1), strict=True):
        assert abs(len(manifest.of(name)) / total - target) < 0.03
    by_id = {r.record_id: r for r in records}
    positives = [e for e in manifest.of("test") if by_id[e.record_id].codes_present()]
    assert positives, "the test split must contain positives"


def test_a_source_can_ask_for_a_bigger_test_share() -> None:
    records, emb, hashes = _mouths(300, 4)
    records = [
        r.model_copy(update={"source": "alphadent" if i % 8 < 4 else "aku_caries"})
        for i, r in enumerate(records)
    ]
    groups = group(records, emb, hashes, 0.05, 2)
    manifest = split(records, groups, 3, source_ratios={"alphadent": (0.5, 0.1, 0.4)})
    share = {
        name: len([e for e in manifest.of("test") if e.source == name])
        / len([e for e in manifest.entries if e.source == name])
        for name in ("alphadent", "aku_caries")
    }
    assert abs(share["alphadent"] - 0.4) < 0.05
    assert abs(share["aku_caries"] - 0.1) < 0.05


def test_adding_a_source_does_not_move_images_that_were_already_placed() -> None:
    """Results on a split stay valid when a new dataset joins: splits are append-only."""
    old_records = [_record(i, patient=f"p{i // 3}", pos=i % 4 == 0) for i in range(240)]
    new_records = [
        _record(i, source="alphadent", patient=f"a{i // 2}", pos=i % 3 == 0) for i in range(100)
    ]
    old_groups = [f"p:{r.patient_id}" for r in old_records]
    new_groups = [f"p:{r.patient_id}" for r in new_records]
    before = split(old_records, old_groups, seed=7)
    after = split(old_records + new_records, old_groups + new_groups, seed=7)
    kept = {e.record_id: e.split for e in before.entries}
    assert {e.record_id: e.split for e in after.entries if e.record_id in kept} == kept


def test_split_rejects_bad_ratios() -> None:
    records, groups, _ = _manifest()
    with pytest.raises(ValueError, match="sum to 1"):
        split(records, groups, 1, (0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="sum to 1"):
        split(records, groups, 1, source_ratios={"alphadent": (0.5, 0.2, 0.2)})
