"""Strict parsing of a model answer into a VisionOutput, then the release decision on top.

No repair: an answer that is not the agreed JSON is a ParseError, counted by the evaluation as a
format failure. The one normalisation is stripping a single markdown fence, applied to every model
alike (zero-shot baselines like to add it).
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dentabridge_core import Finding, FindingCode, Region, VisionOutput
from dentabridge_core.analysis import ImageQuality, QualityReason

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class ParseError(ValueError):
    pass


class ReleaseConfig(BaseModel):
    """What the shipped weights may claim. Written next to the weights by the release gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessed: frozenset[FindingCode]  # codes allowed to report possible / not_seen
    thresholds: Mapping[FindingCode, float] = Field(default_factory=dict)
    needs_scores: bool = False

    @classmethod
    def read(cls, path: Path) -> "ReleaseConfig":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class _Region(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x0: float
    y0: float
    x1: float
    y1: float


class _Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: FindingCode
    status: Literal["possible", "not_seen"]
    regions: list[_Region] = Field(default_factory=list)


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_quality: ImageQuality
    quality_reason: QualityReason | None = None
    findings: list[_Finding] = Field(default_factory=list)


def _load(raw: str) -> _Answer:
    text = raw.strip()
    if match := _FENCE.match(text):
        text = match.group(1)
    try:
        return _Answer.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as error:
        raise ParseError(f"not the agreed JSON: {error}") from error


def parse_output(
    raw: str,
    token_scores: Mapping[FindingCode, float] | None,
    release: ReleaseConfig,
    latency_ms: float = 0.0,
) -> VisionOutput:
    if release.needs_scores and token_scores is None:
        raise ParseError("this release needs token scores and the backend gave none")
    answer = _load(raw)
    try:
        return VisionOutput(
            image_quality=answer.image_quality,
            quality_reason=answer.quality_reason,
            findings=tuple(_finding(item, token_scores, release) for item in answer.findings),
            latency_ms=latency_ms,
        )
    except ValidationError as error:
        raise ParseError(f"answer breaks the output contract: {error}") from error


def _finding(
    item: _Finding, token_scores: Mapping[FindingCode, float] | None, release: ReleaseConfig
) -> Finding:
    if item.code not in release.assessed:
        return Finding(code=item.code, status="not_assessed")
    score = None if token_scores is None else token_scores.get(item.code)
    status = item.status
    if score is not None and item.code in release.thresholds:
        status = "possible" if score >= release.thresholds[item.code] else "not_seen"
    regions = tuple(Region(**r.model_dump()) for r in item.regions) if status == "possible" else ()
    return Finding(code=item.code, status=status, score=score, regions=regions)
