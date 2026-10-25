"""The demo urgency hint. It comes from a rule table, never from the model."""

from typing import Literal

from dentabridge_core._base import Contract
from dentabridge_core.findings import FindingCode

UrgencyLevel = Literal["routine", "soon", "prompt", "retake_photo"]
ReviewStatus = Literal["illustrative_not_clinically_reviewed", "reviewed"]


class UrgencyHint(Contract):
    level: UrgencyLevel
    rule_id: str
    table_version: str
    review_status: ReviewStatus
    triggered_by: tuple[FindingCode, ...] = ()
