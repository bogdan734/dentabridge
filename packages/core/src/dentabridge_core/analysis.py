"""The result of one photo analysis, as the API returns it."""

from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, model_validator

from dentabridge_core._base import Contract, Locale
from dentabridge_core.findings import ASSESSABLE, Finding, FindingCode, Modality, ModelInfo
from dentabridge_core.urgency import UrgencyHint

ImageQuality = Literal["ok", "unusable"]
QualityReason = Literal["blurry", "dark", "no_teeth", "not_dental"]
NoteStatus = Literal["absent", "passed", "rejected"]


class PatientText(Contract):
    locale: Locale
    observations: tuple[str, ...]
    urgency_line: str
    see_dentist_line: str
    not_assessed_line: str
    disclaimer: str


class AnalysisResult(Contract):
    analysis_id: UUID
    modality: Modality
    model: ModelInfo
    image_quality: ImageQuality
    quality_reason: QualityReason | None = None
    findings: tuple[Finding, ...]
    urgency: UrgencyHint
    patient_text: PatientText
    vlm_note: str | None = None
    vlm_note_status: NoteStatus = "absent"
    created_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        codes = [f.code for f in self.findings]
        if len(codes) != len(FindingCode) or set(codes) != set(FindingCode):
            raise ValueError("findings must list every catalogue code exactly once")
        assessable = ASSESSABLE[self.modality]
        for finding in self.findings:
            if finding.code not in assessable and finding.status != "not_assessed":
                raise ValueError(f"{finding.code} cannot be assessed on {self.modality}")
            if self.image_quality == "unusable" and finding.status != "not_assessed":
                raise ValueError("an unusable image leaves every finding not_assessed")
        if (self.image_quality == "unusable") != (self.quality_reason is not None):
            raise ValueError("quality_reason is set exactly when the image is unusable")
        if (self.vlm_note_status == "passed") != (self.vlm_note is not None):
            raise ValueError("vlm_note is present exactly when its status is passed")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self
