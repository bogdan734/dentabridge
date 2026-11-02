"""Training targets: the JSON answer a record should produce for the codes asked about."""

import json

from dentabridge_core import FindingCode
from dentabridge_core.analysis import QualityReason
from dentabridge_ml.data.records import ImageRecord, LabelBox
from dentabridge_ml.data.sources import SOURCES

SCALE = 1000  # boxes are integers in thousandths of the image side: the scale Qwen3-VL grounds in
MAX_REGIONS = 6  # the largest boxes per finding: the answer must stay short enough to train on


def _region(box: LabelBox) -> list[int] | None:
    """[x0, y0, x1, y1] in thousandths, or None when rounding leaves no area."""
    x0, y0, x1, y1 = (round(v * SCALE) for v in (box.x0, box.y0, box.x1, box.y1))
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def _largest(boxes: list[LabelBox]) -> list[LabelBox]:
    drawable = [b for b in boxes if _region(b) is not None]
    ranked = sorted(drawable, key=lambda b: (b.x1 - b.x0) * (b.y1 - b.y0), reverse=True)
    return sorted(ranked[:MAX_REGIONS], key=lambda b: (b.y0, b.x0))


def target_json(rec: ImageRecord, assess: frozenset[FindingCode]) -> str:
    """The answer for a usable image. Only `assess` is judged, and every code in it must be one
    the record's source labels: an unlabelled code must never be trained as "not_seen"."""
    unlabelled = assess - SOURCES[rec.source].coverage
    if unlabelled:
        raise ValueError(f"{rec.record_id}: no labels for {sorted(unlabelled)}")
    findings = []
    for code in FindingCode:
        if code not in assess:
            continue
        boxes = [b for b in rec.boxes if b.code == code]
        present = bool(boxes) or code in rec.image_labels
        findings.append(
            {
                "code": code.value,
                "status": "possible" if present else "not_seen",
                "regions": [_region(b) for b in _largest(boxes)],  # all drawable
            }
        )
    answer: dict[str, object] = {
        "image_quality": "ok",
        "quality_reason": None,
        "findings": findings,
    }
    return json.dumps(answer, separators=(",", ":"))


def unusable_json(reason: QualityReason) -> str:
    answer: dict[str, object] = {
        "image_quality": "unusable",
        "quality_reason": reason,
        "findings": [],
    }
    return json.dumps(answer, separators=(",", ":"))
