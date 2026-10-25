"""`python -m dentabridge_ml.eval run`: answer a split with one backend, then score the answers."""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from dentabridge_ml.data.loaders import load_records
from dentabridge_ml.data.sources import default_root
from dentabridge_ml.eval.predict import predict, read_predictions, research_release
from dentabridge_ml.eval.report import build_result
from dentabridge_ml.eval.results import render_markdown
from dentabridge_ml.eval.slices import slice_results
from dentabridge_ml.serving import load_vision_backend
from dentabridge_ml.splits.build import MANIFEST_DIR
from dentabridge_ml.splits.manifest import read_entries


def _run(args: argparse.Namespace) -> int:
    entries = read_entries(args.manifests, args.split)[: args.limit]
    sources = sorted({e.source for e in entries})
    records = {r.record_id: r for s in sources for r in load_records(s, args.root)}
    backend = load_vision_backend(args.backend, research_release(), args.image_max_px)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.out_dir / f"{args.split}.predictions.jsonl"
    asyncio.run(
        predict(
            backend,
            list(records.values()),
            entries,
            args.root,
            predictions,
            final=args.final,
            access_log=args.access_log,
            batch_size=args.batch_size,
        )
    )
    rows = read_predictions(predictions)
    model_id = backend.model.model_id
    result = build_result(model_id, args.split, records, entries, rows)
    for by in ("source", "view"):
        sliced = slice_results(model_id, args.split, records, entries, rows, by=by)
        text = "\n".join(render_markdown(part) for part in sliced.values())
        (args.out_dir / f"{args.split}.by-{by}.md").write_text(text, encoding="utf-8")
    result.write(args.out_dir / f"{args.split}.json")
    table = render_markdown(result)
    (args.out_dir / f"{args.split}.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dentabridge_ml.eval")
    steps = parser.add_subparsers(dest="step", required=True)
    run = steps.add_parser("run", help="predict one split with one backend and score it")
    run.add_argument("--backend", required=True, help="ollama:<model> or transformers:<directory>")
    run.add_argument("--split", choices=("val", "test"), required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--image-max-px", type=int, default=None)
    run.add_argument(
        "--limit", type=int, default=None, help="first N images; val only (smoke runs)"
    )
    run.add_argument(
        "--batch-size", type=int, default=1, help="pictures per forward pass (transformers)"
    )
    run.add_argument("--final", action="store_true", help="read the test split, once, and log it")
    run.add_argument("--access-log", type=Path, default=None)
    run.add_argument("--manifests", type=Path, default=MANIFEST_DIR)
    run.add_argument("--root", type=Path, default=default_root())
    run.set_defaults(func=_run)
    args = parser.parse_args(argv)
    if args.split == "test" and args.limit is not None:
        parser.error("the test split is scored whole: --limit is for val smoke runs")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
