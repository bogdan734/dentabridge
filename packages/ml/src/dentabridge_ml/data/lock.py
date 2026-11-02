"""data.lock.json: one tree digest per unpacked source, so tampered or partial data is caught.

The digest covers every file path, size and sha256 under <root>/<source>/. The lock is small
enough to commit; the per-file listing it is built from stays local.
"""

import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from dentabridge_ml.data.download import file_digest, source_dir
from dentabridge_ml.data.sources import SOURCES, SourceId

LOCK_VERSION = 1


class LockMismatch(Exception):
    pass


@dataclass(frozen=True)
class TreeDigest:
    sha256: str
    files: int
    bytes: int


def tree_digest(directory: Path) -> TreeDigest:
    paths = sorted(p for p in directory.rglob("*") if p.is_file())
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = list(pool.map(lambda p: file_digest(p, "sha256"), paths))
    outer = hashlib.sha256()
    total = 0
    for path, digest in zip(paths, digests, strict=True):
        size = path.stat().st_size
        total += size
        outer.update(f"{path.relative_to(directory).as_posix()}\0{size}\0{digest}\n".encode())
    return TreeDigest(sha256=outer.hexdigest(), files=len(paths), bytes=total)


def write_lock(root: Path, lock: Path, sources: list[SourceId], today: date | None = None) -> None:
    """Record the digest of each source; existing entries of other sources are kept."""
    document: dict[str, Any] = (
        json.loads(lock.read_text()) if lock.exists() else {"version": LOCK_VERSION}
    )
    entries: dict[str, Any] = document.setdefault("sources", {})
    for source in sources:
        if not SOURCES[source].published:
            raise ValueError(f"{source} is private and never enters the lock")
        entries[source] = {
            **asdict(tree_digest(source_dir(root, source))),
            "licence": SOURCES[source].licence,
            "locked_on": (today or date.today()).isoformat(),
        }
    lock.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def verify_lock(root: Path, lock: Path, sources: Sequence[SourceId] | None = None) -> None:
    """Every locked source, or only `sources` (a machine that holds some of the datasets)."""
    document = json.loads(lock.read_text())
    problems: list[str] = []
    names: Sequence[SourceId] = sources or list(document["sources"])
    for source in names:
        entry = document["sources"].get(source)
        if entry is None:
            problems.append(f"{source}: not in the lock")
            continue
        directory = source_dir(root, source)
        if not directory.is_dir():
            problems.append(f"{source}: missing directory {directory}")
            continue
        actual = tree_digest(directory)
        if actual.sha256 != entry["sha256"]:
            problems.append(
                f"{source}: tree digest differs "
                f"(locked {entry['files']} files / {entry['bytes']} bytes, "
                f"found {actual.files} / {actual.bytes})"
            )
    if problems:
        raise LockMismatch("; ".join(problems))
