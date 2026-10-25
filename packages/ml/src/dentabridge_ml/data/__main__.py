"""`python -m dentabridge_ml.data fetch|extract|lock|verify [SOURCE...]`, used by `make data`."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from dentabridge_ml.data.download import download, extract, source_dir
from dentabridge_ml.data.lock import verify_lock, write_lock
from dentabridge_ml.data.sources import SOURCES, SourceId, default_root

LOCK_PATH = Path(__file__).resolve().parents[3] / "data.lock.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dentabridge_ml.data", description=__doc__)
    parser.add_argument("action", choices=["fetch", "extract", "lock", "verify"])
    parser.add_argument("sources", nargs="*", choices=get_args(SourceId))
    parser.add_argument("--root", type=Path, default=default_root())
    args = parser.parse_args(argv)
    chosen: list[SourceId] = args.sources or [s for s in SOURCES if SOURCES[s].published]
    if args.action == "verify":
        verify_lock(args.root, LOCK_PATH)
    elif args.action == "lock":
        write_lock(args.root, LOCK_PATH, chosen)
    elif args.action == "fetch":
        for source in chosen:
            print(source, download(source, args.root))
    else:
        for source in chosen:
            target = source_dir(args.root, source)
            if target.exists() and any(target.iterdir()):
                print(source, "already extracted")
            else:
                print(source, extract(source, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
