"""`python -m dentabridge_ml.splits.build`: records -> groups -> manifests, and a plain summary."""

import argparse
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import get_args

from dentabridge_ml.data.loaders import LOADERS, load_records
from dentabridge_ml.data.records import ImageRecord
from dentabridge_ml.data.sources import SOURCES, SourceId, default_root
from dentabridge_ml.splits.group import group, phash
from dentabridge_ml.splits.manifest import SPLITS, SplitManifest, split, write_manifests

MANIFEST_DIR = Path(__file__).resolve().parents[3] / "manifests"
MAX_HAMMING = 2  # dHash distance for "the same photo"; teeth of different people sit at 3 and up
MAX_COS_DIST = 0.05
# AlphaDent is small: 10% would leave about 22 crown positives, under the gate's minimum of 30.
SOURCE_RATIOS: dict[str, tuple[float, float, float]] = {"alphadent": (0.7, 0.1, 0.2)}


def build(
    root: Path, sources: Sequence[SourceId], seed: int
) -> tuple[list[ImageRecord], SplitManifest]:
    records = [r for s in sources for r in load_records(s, root)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(lambda r: phash(root / r.source / r.path), records))
    if any(r.patient_id is None and r.capture_sequence is None for r in records):
        raise NotImplementedError("embeddings for records without an id are not wired in yet")
    groups = group(records, None, hashes, MAX_COS_DIST, MAX_HAMMING)
    return records, split(records, groups, seed, source_ratios=SOURCE_RATIOS)


def summary(records: Sequence[ImageRecord], manifest: SplitManifest) -> str:
    by_id = {r.record_id: r for r in records}
    sizes = Counter(e.group for e in manifest.entries)
    lines = [f"{len(records)} images in {len(sizes)} groups; largest group {max(sizes.values())}"]
    for name in SPLITS:
        entries = manifest.of(name)
        positives = sum(bool(by_id[e.record_id].codes_present()) for e in entries)
        groups = len({e.group for e in entries})
        lines.append(f"{name}: {len(entries)} images, {groups} groups, {positives} positive")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dentabridge_ml.splits.build")
    parser.add_argument("sources", nargs="*", choices=get_args(SourceId))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args(argv)
    chosen: list[SourceId] = args.sources or [s for s in LOADERS if SOURCES[s].published]
    records, manifest = build(args.root, chosen, args.seed)
    write_manifests(manifest, MANIFEST_DIR)
    print(summary(records, manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
