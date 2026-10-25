"""Training targets: the JSON answer a record should produce for the codes asked about."""

import json

from dentabridge_core import FindingCode
from dentabridge_core.analysis import QualityReason
from dentabridge_ml.data.records import ImageRecord, LabelBox
from dentabridge_ml.data.sources import SOURCES

PRECISION = 3
MAX_REGIONS = 6  # the largest boxes per finding: the answer must stay short enough to train on


def _region(box: LabelBox) -> dict[str, float]:
    return {
        "x0": round(box.x0, PRECISION),
        "y0": round(box.y0, PRECISION),
        "x1": round(box.x1, PRECISION),
        "y1": round(box.y1, PRECISION),
    }


def _largest(boxes: list[LabelBox]) -> list[LabelBox]:
    ranked = sorted(boxes, key=lambda b: (b.x1 - b.x0) * (b.y1 - b.y0), reverse=True)
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
                "regions": [_region(b) for b in _largest(boxes)],
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
