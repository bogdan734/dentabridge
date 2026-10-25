import asyncio
from datetime import date

import pytest
from pydantic import ValidationError

from dentabridge_core import (
    FareEstimate,
    FareEstimator,
    FareRequest,
    Finding,
    FindingCode,
    GlossaryEntry,
    Money,
    PriceRange,
    SanitisedImage,
    Translator,
    VisionBackend,
    VisionOutput,
)
from dentabridge_core.testing import (
    FakeFareEstimator,
    FakeTranslator,
    FakeVisionBackend,
    sample_findings,
)

IMAGE = SanitisedImage(data=b"jpeg", width=2, height=2, sha256="a" * 64)
OUTPUT = VisionOutput(image_quality="ok", findings=sample_findings(), latency_ms=12.5)


def test_fakes_satisfy_ports() -> None:
    assert isinstance(FakeVisionBackend({}), VisionBackend)
    assert isinstance(FakeFareEstimator(), FareEstimator)
    assert isinstance(FakeTranslator("> "), Translator)
    assert not isinstance(object(), VisionBackend)


def test_fake_vision_backend_answers_by_hash_and_fails_loudly() -> None:
    backend = FakeVisionBackend({IMAGE.sha256: OUTPUT})
    assert asyncio.run(backend.analyse(IMAGE, "intraoral_photo")) == OUTPUT
    other = IMAGE.model_copy(update={"sha256": "b" * 64})
    with pytest.raises(KeyError):
        asyncio.run(backend.analyse(other, "intraoral_photo"))


def test_fake_fare_estimator_scales_with_travellers() -> None:
    def estimate(travellers: int) -> FareEstimate:
        request = FareRequest(
            origin_iata="LHR",
            dest_iata="AAA",
            depart=date(2026, 12, 1),
            return_=date(2026, 12, 8),
            booked_on=date(2026, 11, 1),
            travellers=travellers,
        )
        return FakeFareEstimator().estimate(request)

    assert estimate(3).range_usd.mid.amount_minor == 3 * estimate(1).range_usd.mid.amount_minor


def test_fake_translator_prefixes_and_ignores_glossary() -> None:
    translated = asyncio.run(
        FakeTranslator("> ").translate("hello", "en", "ru", [GlossaryEntry(src="a", tgt="b")])
    )
    assert translated.text == "> hello"


def test_fare_request_and_estimate_rules() -> None:
    with pytest.raises(ValidationError, match="return_"):
        FareRequest(
            origin_iata="LHR",
            dest_iata="AAA",
            depart=date(2026, 12, 8),
            return_=date(2026, 12, 1),
            booked_on=date(2026, 11, 1),
            travellers=1,
        )
    eur = Money(amount_minor=100, currency="EUR")
    with pytest.raises(ValidationError, match="USD"):
        FareEstimate(
            range_usd=PriceRange(low=eur, mid=eur, high=eur),
            interval="q20-q80",
            international=False,
            basis_key="x",
        )


def test_vision_output_rules() -> None:
    caries = Finding(code=FindingCode.CARIES, status="not_seen")
    with pytest.raises(ValidationError, match="appears twice"):
        VisionOutput(image_quality="ok", findings=(caries, caries), latency_ms=1.0)
    with pytest.raises(ValidationError, match="quality_reason"):
        VisionOutput(image_quality="unusable", findings=(), latency_ms=1.0)
    with pytest.raises(ValidationError):
        VisionOutput(image_quality="ok", findings=(), latency_ms=-1.0)


def test_sanitised_image_needs_a_real_hash() -> None:
    with pytest.raises(ValidationError):
        SanitisedImage(data=b"x", width=1, height=1, sha256="not-a-hash")
