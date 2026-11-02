"""One analysis: the backend's observations, the demo urgency rules, then the patient text."""

from datetime import timedelta

from dentabridge_api.clock import Clock
from dentabridge_api.ids import new_id
from dentabridge_core import (
    AnalysisResult,
    Finding,
    FindingCode,
    Locale,
    Modality,
    SanitisedImage,
    VisionBackend,
)
from dentabridge_core.policy import (
    UrgencyTable,
    evaluate,
    guard_note,
    render_patient_text,
)


def complete(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    """Every catalogue code exactly once: what the model did not judge is not_assessed."""
    seen = {f.code: f for f in findings}
    return tuple(seen.get(code, Finding(code=code, status="not_assessed")) for code in FindingCode)


async def run_analysis(
    image: SanitisedImage,
    modality: Modality,
    locale: Locale,
    backend: VisionBackend,
    clock: Clock,
    ttl: timedelta,
    table: UrgencyTable,
) -> AnalysisResult:
    output = await backend.analyse(image, modality)
    # a finding the modality cannot judge is not_assessed whatever the backend said
    findings = complete(output.findings)
    if output.image_quality == "unusable":
        findings = tuple(Finding(code=f.code, status="not_assessed") for f in findings)
    urgency = evaluate(findings, output.image_quality, table)
    note, note_status = guard_note(output.note_en)
    now = clock.now()
    return AnalysisResult(
        analysis_id=new_id(),
        modality=modality,
        model=backend.model,
        image_quality=output.image_quality,
        quality_reason=output.quality_reason,
        findings=findings,
        urgency=urgency,
        patient_text=render_patient_text(findings, urgency, locale, modality),
        vlm_note=note,
        vlm_note_status=note_status,
        created_at=now,
        expires_at=now + ttl,
    )
