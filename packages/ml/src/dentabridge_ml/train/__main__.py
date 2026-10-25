"""`python -m dentabridge_ml.train sft|run|merge`: the steps of one fine-tune, in order.

`sft` needs no GPU stack; `run` and `merge` import torch only when they are asked for.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from dentabridge_ml.data.loaders import load_records
from dentabridge_ml.data.sources import default_root
from dentabridge_ml.splits.build import MANIFEST_DIR
from dentabridge_ml.splits.manifest import read_entries
from dentabridge_ml.train.config import load_config
from dentabridge_ml.train.data import build_rows, write_sft


def _sft(args: argparse.Namespace) -> int:
    entries = read_entries(args.manifests, args.split)
    sources = sorted({e.source for e in entries})
    records = {r.record_id: r for s in sources for r in load_records(s, args.root)}
    rows = build_rows(records, entries, args.negatives, args.seed)[: args.limit]
    print(f"{write_sft(rows, args.out)} rows -> {args.out}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from dentabridge_ml.train.run import train

    adapter = train(load_config(args.config), args.rows, args.root, args.out, args.resume)
    print(f"adapter -> {adapter}")
    return 0


def _merge(args: argparse.Namespace) -> int:
    from dentabridge_ml.train.merge import merge_adapter

    print(f"merged -> {merge_adapter(args.base, args.adapter, args.out, args.dtype)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dentabridge_ml.train")
    steps = parser.add_subparsers(dest="step", required=True)

    sft = steps.add_parser("sft", help="write the training rows of a split as JSON lines")
    sft.add_argument("split", choices=("train", "val"), help="the test split never trains")
    sft.add_argument("--out", type=Path, required=True)
    sft.add_argument("--manifests", type=Path, default=MANIFEST_DIR)
    sft.add_argument("--root", type=Path, default=default_root())
    sft.add_argument("--negatives", type=float, default=0.15, help="degraded photos per real one")
    sft.add_argument("--seed", type=int, default=0)
    sft.add_argument("--limit", type=int, default=None, help="keep the first N rows (pilots)")
    sft.set_defaults(func=_sft)

    run = steps.add_parser("run", help="fine-tune with LoRA")
    run.add_argument("config", type=Path)
    run.add_argument("rows", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--root", type=Path, default=default_root())
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=_run)

    merge = steps.add_parser("merge", help="fold an adapter into the base weights")
    merge.add_argument("adapter", type=Path)
    merge.add_argument("--base", required=True, help="the id or directory the adapter was tuned on")
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    merge.set_defaults(func=_merge)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
