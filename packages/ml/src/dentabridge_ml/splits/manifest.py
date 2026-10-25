"""Split manifests: which record is in which split, decided once, by group, from a seed."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dentabridge_ml.data.records import ImageRecord
from dentabridge_ml.splits.group import GroupMethod, group_method

SplitName = Literal["train", "val", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "val", "test")


class SplitEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    record_id: str
    source: str
    split: SplitName
    group: str
    group_method: GroupMethod


class SplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seed: int
    ratios: tuple[float, float, float]
    entries: tuple[SplitEntry, ...]

    def of(self, split: SplitName) -> tuple[SplitEntry, ...]:
        return tuple(e for e in self.entries if e.split == split)

    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.entries).encode()).hexdigest()


def _canonical(entries: Sequence[SplitEntry]) -> str:
    rows = sorted(entries, key=lambda e: e.record_id)
    return "".join(json.dumps(e.model_dump(), sort_keys=True) + "\n" for e in rows)


def _stratum(record: ImageRecord) -> str:
    return f"{record.source}:{'pos' if record.codes_present() else 'neg'}"


def _most_underfilled(
    stratum: str,
    size: int,
    assigned: Mapping[tuple[str, SplitName], int],
    stratum_total: int,
    ratios: tuple[float, float, float],
) -> SplitName:
    """The split that stays closest to its target share of this stratum after taking the group."""

    def fill(name: SplitName) -> float:
        return (assigned[(stratum, name)] + size) / (ratios[SPLITS.index(name)] * stratum_total)

    return min(SPLITS, key=fill)


def split(
    records: Sequence[ImageRecord],
    groups: Sequence[str],
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> SplitManifest:
    """Assign whole groups to splits, balancing each source x positive/negative stratum."""
    if len(records) != len(groups) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("records and groups must line up and the ratios must sum to 1")
    members: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(groups):
        members[name].append(index)
    strata = Counter(_stratum(r) for r in records)
    group_stratum = {
        name: Counter(_stratum(records[i]) for i in idx).most_common(1)[0][0]
        for name, idx in members.items()
    }
    assigned: dict[tuple[str, SplitName], int] = Counter()
    order = sorted(members)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda name: -len(members[name]))  # stable: big groups first, seeded ties
    placement: dict[str, SplitName] = {}
    for name in order:
        stratum, size = group_stratum[name], len(members[name])
        placement[name] = _most_underfilled(stratum, size, assigned, strata[stratum], ratios)
        assigned[(stratum, placement[name])] += size
    entries = tuple(
        SplitEntry(
            record_id=r.record_id,
            source=r.source,
            split=placement[groups[i]],
            group=groups[i],
            group_method=group_method(r),
        )
        for i, r in enumerate(records)
    )
    return SplitManifest(seed=seed, ratios=ratios, entries=entries)


def write_manifests(manifest: SplitManifest, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        (directory / f"{name}.jsonl").write_text(_canonical(manifest.of(name)))
    lines = [
        f"{hashlib.sha256((directory / f'{n}.jsonl').read_bytes()).hexdigest()}  {n}.jsonl\n"
        for n in SPLITS
    ]
    (directory / "manifests.sha256").write_text("".join(lines))
