"""Fakes and builders that every workstream uses in its tests. Not imported by production code."""

from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID

from dentabridge_core._base import Locale
from dentabridge_core.analysis import AnalysisResult, PatientText
from dentabridge_core.findings import (
    ASSESSABLE,
    Finding,
    FindingCode,
    FindingStatus,
    Modality,
    ModelInfo,
    Region,
)
from dentabridge_core.money import Money, PriceRange
from dentabridge_core.ports import (
    FareEstimate,
    FareRequest,
    GlossaryEntry,
    SanitisedImage,
    Translation,
    VisionOutput,
)
from dentabridge_core.urgency import UrgencyHint


class FakeVisionBackend:
    """Answers from a table keyed by image sha256; an unknown image is a test bug, so KeyError."""

    def __init__(self, outputs: Mapping[str, VisionOutput], model: ModelInfo | None = None) -> None:
        self.model = model or ModelInfo(model_id="fake-vision", version="0", backend="fixture")
        self._outputs = dict(outputs)

    async def analyse(self, image: SanitisedImage, modality: Modality) -> VisionOutput:
        return self._outputs[image.sha256]


class FakeFareEstimator:
    """Fixed USD 80 / 100 / 130 per traveller, whatever the route. For tests only."""

    model_version = "fake-1"

    def estimate(self, req: FareRequest) -> FareEstimate:
        def usd(per_traveller: int) -> Money:
            return Money(amount_minor=per_traveller * req.travellers, currency="USD")

        return FareEstimate(
            range_usd=PriceRange(low=usd(8_000), mid=usd(10_000), high=usd(13_000)),
            interval="q20-q80",
            international=False,
            basis_key="fake",
        )


class FakeTranslator:
    def __init__(self, prefix: str) -> None:
        self.model_id = "fake-translator"
        self._prefix = prefix

    async def translate(
        self, text: str, src: str, tgt: str, glossary: Sequence[GlossaryEntry]
    ) -> Translation:
        return Translation(text=f"{self._prefix}{text}", latency_ms=0.0)


def sample_findings(
    possible: Collection[FindingCode] = (), modality: Modality = "intraoral_photo"
) -> tuple[Finding, ...]:
    """Every catalogue code once: `possible` ones flagged, other assessable ones not_seen."""

    def build(code: FindingCode) -> Finding:
        if code not in ASSESSABLE[modality]:
            return Finding(code=code, status="not_assessed")
        if code in possible:
            return Finding(
                code=code,
                status="possible",
                score=0.9,
                regions=(Region(x0=0.1, y0=0.1, x1=0.4, y1=0.4),),
            )
        status: FindingStatus = "not_seen"
        return Finding(code=code, status=status, score=0.1)

    return tuple(build(code) for code in FindingCode)


def sample_analysis(
    possible: Collection[FindingCode] = (),
    *,
    modality: Modality = "intraoral_photo",
    locale: Locale = "en",
) -> AnalysisResult:
    created = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
    return AnalysisResult(
        analysis_id=UUID(int=1),
        modality=modality,
        model=ModelInfo(model_id="fake-vision", version="0", backend="fixture"),
        image_quality="ok",
        findings=sample_findings(possible, modality),
        urgency=UrgencyHint(
            level="routine",
            rule_id="R9",
            table_version="test",
            review_status="illustrative_not_clinically_reviewed",
        ),
        patient_text=PatientText(
            locale=locale,
            observations=(),
            urgency_line="test",
            see_dentist_line="test",
            not_assessed_line="test",
            disclaimer="test",
        ),
        created_at=created,
        expires_at=created + timedelta(hours=24),
    )
