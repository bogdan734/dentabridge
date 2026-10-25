"""Demo urgency: a deterministic rule table over validated findings, never the model's opinion."""

import hashlib
import tomllib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from dentabridge_core.analysis import ImageQuality
from dentabridge_core.findings import Finding, FindingCode, FindingStatus
from dentabridge_core.urgency import ReviewStatus, UrgencyHint, UrgencyLevel

DEFAULT_TABLE_PATH = Path(__file__).with_name("urgency_v1.toml")
ILLUSTRATIVE: ReviewStatus = "illustrative_not_clinically_reviewed"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UrgencyRule(_Strict):
    id: str
    level: UrgencyLevel
    enabled: bool = True
    quality: ImageQuality | None = None
    code: FindingCode | None = None
    status: FindingStatus | None = None
    subtype_in: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _conditions(self) -> Self:
        if self.code is None and (self.status is not None or self.subtype_in is not None):
            raise ValueError(f"rule {self.id}: status and subtype_in need a code")
        if self.enabled and self.subtype_in is not None and not self.subtype_in:
            raise ValueError(f"rule {self.id}: an enabled subtype rule needs subtypes")
        return self

    def matches(
        self, findings: Sequence[Finding], quality: ImageQuality
    ) -> tuple[FindingCode, ...] | None:
        """The codes that made the rule fire (none for a quality or catch-all rule), or None."""
        if self.quality is not None and self.quality != quality:
            return None
        if self.code is None:
            return ()
        hits = tuple(
            f.code
            for f in findings
            if f.code == self.code
            and (self.status is None or f.status == self.status)
            and (not self.subtype_in or f.subtype in self.subtype_in)
        )
        return hits or None


class _RawTable(_Strict):
    version: str
    rule: tuple[UrgencyRule, ...]


class UrgencyTable(_Strict):
    version: str
    rules: tuple[UrgencyRule, ...]
    sha256: str
    review_status: ReviewStatus


class RuleVerdict(_Strict):
    verdict: Literal["agree", "change", "remove"]
    comment: str = ""


class ReviewRecord(_Strict):
    """A dentist's signed review of one exact table file (see docs/clinical-review/)."""

    reviewer_name: str
    qualification: str
    registration: str
    reviewed_on: date
    table_sha256: str
    verdicts: dict[str, RuleVerdict]
    statement: str


def _reviewed(sha256: str, rule_ids: set[str], reviews_dir: Path) -> bool:
    for path in sorted(reviews_dir.glob("*.json")):
        record = ReviewRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if (
            record.table_sha256 == sha256
            and rule_ids <= set(record.verdicts)
            and all(v.verdict == "agree" for v in record.verdicts.values())
        ):
            return True
    return False


def load_table(path: Path, reviews_dir: Path | None = None) -> UrgencyTable:
    raw_bytes = path.read_bytes()
    raw = _RawTable.model_validate(tomllib.loads(raw_bytes.decode("utf-8")))
    ids = [r.id for r in raw.rule]
    if len(ids) != len(set(ids)):
        raise ValueError("rule ids must be unique")
    enabled = [r for r in raw.rule if r.enabled]
    if not enabled or enabled[-1].code is not None or enabled[-1].quality is not None:
        raise ValueError("the last enabled rule must match everything")
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    reviewed = reviews_dir is not None and _reviewed(sha256, {r.id for r in enabled}, reviews_dir)
    return UrgencyTable(
        version=raw.version,
        rules=raw.rule,
        sha256=sha256,
        review_status="reviewed" if reviewed else ILLUSTRATIVE,
    )


def review_status(table_path: Path, reviews_dir: Path) -> ReviewStatus:
    return load_table(table_path, reviews_dir).review_status


def evaluate(
    findings: Sequence[Finding], quality: ImageQuality, table: UrgencyTable
) -> UrgencyHint:
    for rule in table.rules:
        if not rule.enabled:
            continue
        triggered = rule.matches(findings, quality)
        if triggered is not None:
            return UrgencyHint(
                level=rule.level,
                rule_id=rule.id,
                table_version=table.version,
                review_status=table.review_status,
                triggered_by=triggered,
            )
    raise RuntimeError("urgency table has no catch-all rule")  # load_table rules this out
