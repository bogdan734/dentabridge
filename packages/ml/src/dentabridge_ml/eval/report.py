"""Turn predictions into per-finding numbers.

A finding is scored only on images whose source labels it, so a code a source never labelled is
neither a hit nor a miss there. An answer that was not valid JSON counts as "not possible":
the conservative reading, reported next to the valid-answer rate rather than hidden by it.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from dentabridge_core import ASSESSABLE, FindingCode
from dentabridge_ml.data.records import ImageRecord
from dentabridge_ml.data.sources import SOURCES
from dentabridge_ml.eval.calibration import ece, risk_coverage
from dentabridge_ml.eval.metrics import binary_stats
from dentabridge_ml.eval.predict import PredictionRow
from dentabridge_ml.eval.results import CodeResult, SplitResult
from dentabridge_ml.splits.manifest import SplitManifest, SplitName


def _possible(row: PredictionRow, code: FindingCode) -> tuple[bool, float | None]:
    if row.output is None:
        return False, None
    for finding in row.output.findings:
        if finding.code == code:
            return finding.status == "possible", finding.score
    return False, None


def build_result(
    model_id: str,
    split: SplitName,
    records: Mapping[str, ImageRecord],
    manifest: SplitManifest,
    predictions: Sequence[PredictionRow],
    seed: int = 0,
) -> SplitResult:
    group_of = {e.record_id: e.group for e in manifest.entries}
    codes: list[CodeResult] = []
    for code in sorted(ASSESSABLE["intraoral_photo"], key=list(FindingCode).index):
        rows = [
            (p, records[p.record_id])
            for p in predictions
            if code in SOURCES[records[p.record_id].source].coverage
        ]
        if not rows:
            codes.append(CodeResult(code=code, n_pos=0, n_neg=0))
            continue
        labels = [code in rec.codes_present() for _, rec in rows]
        decided = [_possible(p, code) for p, _ in rows]
        preds = [pred for pred, _ in decided]
        stats = binary_stats(preds, labels, code, [group_of[p.record_id] for p, _ in rows], seed)
        scores = [score for _, score in decided]
        calibrated = all(s is not None for s in scores)
        y = np.array(labels, dtype=bool)
        s_arr = np.array([s for s in scores if s is not None], dtype=np.float64)
        codes.append(
            CodeResult(
                code=code,
                n_pos=sum(labels),
                n_neg=len(labels) - sum(labels),
                sens=stats.sens,
                spec=stats.spec,
                sens_ci=stats.sens_ci,
                spec_ci=stats.spec_ci,
                ece=ece(s_arr, y) if calibrated else None,
                aurc=_aurc(s_arr, y, preds) if calibrated else None,
            )
        )
    valid = sum(p.output is not None for p in predictions)
    return SplitResult(
        model_id=model_id,
        split=split,
        n_images=len(predictions),
        json_valid_rate=valid / len(predictions) if predictions else 0.0,
        codes=tuple(codes),
    )


def _aurc(scores: np.ndarray, labels: np.ndarray, preds: Sequence[bool]) -> float:
    confidence = np.maximum(scores, 1.0 - scores)
    correct = np.array(preds, dtype=bool) == labels
    return risk_coverage(confidence, correct)[2]
