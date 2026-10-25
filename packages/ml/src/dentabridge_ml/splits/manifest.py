"""Split manifests: which record is in which split, decided once, by group, from a seed."""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dentabridge_ml.data.records import ImageRecord
from dentabridge_ml.data.sources import SourceId
from dentabridge_ml.splits.group import GroupMethod, group_method

SplitName = Literal["train", "val", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "val", "test")


class SplitEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    record_id: str
    source: SourceId
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
    source_ratios: Mapping[str, tuple[float, float, float]] | None = None,
) -> SplitManifest:
    """Assign whole groups to splits, balancing each source x positive/negative stratum.

    A small source can ask for a bigger test share through `source_ratios`, so that its findings
    have enough positives to be judged at all.
    """
    overrides = dict(source_ratios or {})
    if len(records) != len(groups) or any(
        abs(sum(r) - 1.0) > 1e-9 for r in (ratios, *overrides.values())
    ):
        raise ValueError("records and groups must line up and every ratio triple must sum to 1")
    members: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(groups):
        members[name].append(index)
    strata = Counter(_stratum(r) for r in records)
    group_stratum = {
        name: Counter(_stratum(records[i]) for i in idx).most_common(1)[0][0]
        for name, idx in members.items()
    }
    assigned: dict[tuple[str, SplitName], int] = Counter()
    # Big groups first, ties in an order that depends only on (seed, group name): the groups of one
    # source are always placed in the same order, so adding a dataset never moves what is placed.
    order = sorted(
        members,
        key=lambda name: (-len(members[name]), hashlib.sha256(f"{seed}:{name}".encode()).digest()),
    )
    placement: dict[str, SplitName] = {}
    for name in order:
        stratum, size = group_stratum[name], len(members[name])
        wanted = overrides.get(stratum.split(":")[0], ratios)
        placement[name] = _most_underfilled(stratum, size, assigned, strata[stratum], wanted)
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


def read_entries(directory: Path, name: SplitName) -> tuple[SplitEntry, ...]:
    """One split's entries, after every manifest file is checked against manifests.sha256."""
    digests = {
        file: digest
        for digest, file in (
            line.split("  ", 1)
            for line in (directory / "manifests.sha256").read_text().splitlines()
        )
    }
    for split_name in SPLITS:
        if f"{split_name}.jsonl" not in digests:
            raise ValueError(f"manifests.sha256 does not list {split_name}.jsonl")
    for file, digest in digests.items():
        if hashlib.sha256((directory / file).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{file} does not match manifests.sha256")
    lines = (directory / f"{name}.jsonl").read_text().splitlines()
    return tuple(SplitEntry.model_validate_json(line) for line in lines if line)


def write_manifests(manifest: SplitManifest, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        (directory / f"{name}.jsonl").write_text(_canonical(manifest.of(name)))
    lines = [
        f"{hashlib.sha256((directory / f'{n}.jsonl').read_bytes()).hexdigest()}  {n}.jsonl\n"
        for n in SPLITS
    ]
    (directory / "manifests.sha256").write_text("".join(lines))
