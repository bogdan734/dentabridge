"""Results per source or per view.

AKU's positives depend heavily on the view (89 % of mandibular photos against 3 % of frontal
ones) and AKU/AlphaDent differ in camera, so a pooled number can hide a model that only learned
the view or the camera. Each slice is scored with the same code as the pooled result.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from dentabridge_ml.data.records import ImageRecord
from dentabridge_ml.eval.predict import PredictionRow
from dentabridge_ml.eval.report import build_result
from dentabridge_ml.eval.results import SplitResult
from dentabridge_ml.splits.manifest import SplitEntry


def slice_results(
    model_id: str,
    split: str,
    records: Mapping[str, ImageRecord],
    entries: Sequence[SplitEntry],
    predictions: Sequence[PredictionRow],
    by: Literal["source", "view"],
) -> dict[str, SplitResult]:
    def name(row: PredictionRow) -> str:
        record = records[row.record_id]
        return record.source if by == "source" else record.view or "unknown"

    parts: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in predictions:
        parts[name(row)].append(row)
    results = {}
    for part, rows in sorted(parts.items()):
        ids = {r.record_id for r in rows}
        subset = [e for e in entries if e.record_id in ids]
        results[part] = build_result(model_id, f"{split} [{by}={part}]", records, subset, rows)
    return results
