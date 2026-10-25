"""The three ports every backend implements, and the data they exchange."""

from collections.abc import Sequence
from datetime import date
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from dentabridge_core._base import Contract, Iata
from dentabridge_core.analysis import ImageQuality, QualityReason
from dentabridge_core.findings import Finding, Modality, ModelInfo
from dentabridge_core.money import PriceRange


class SanitisedImage(Contract):
    """A decoded upload: JPEG bytes without metadata, plus a hash for caching and fixtures."""

    data: bytes
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VisionOutput(Contract):
    image_quality: ImageQuality
    quality_reason: QualityReason | None = None
    findings: tuple[Finding, ...]
    note_en: str | None = None
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        codes = [f.code for f in self.findings]
        if len(codes) != len(set(codes)):
            raise ValueError("a finding code appears twice")
        if (self.image_quality == "unusable") != (self.quality_reason is not None):
            raise ValueError("quality_reason is set exactly when the image is unusable")
        return self


class FareRequest(Contract):
    origin_iata: Iata
    dest_iata: Iata
    depart: date
    return_: date
    booked_on: date
    travellers: int = Field(ge=1, le=6)

    @model_validator(mode="after")
    def _return_not_before_depart(self) -> Self:
        if self.return_ < self.depart:
            raise ValueError("return_ is before depart")
        return self


class FareEstimate(Contract):
    range_usd: PriceRange
    interval: Literal["q20-q80", "q10-q90"]
    international: bool
    basis_key: str

    @model_validator(mode="after")
    def _usd(self) -> Self:
        if self.range_usd.mid.currency != "USD":
            raise ValueError("fare estimates are in USD")
        return self


class GlossaryEntry(Contract):
    src: str = Field(min_length=1)
    tgt: str = Field(min_length=1)


class Translation(Contract):
    text: str
    latency_ms: float = Field(ge=0)


@runtime_checkable
class VisionBackend(Protocol):
    model: ModelInfo

    async def analyse(self, image: SanitisedImage, modality: Modality) -> VisionOutput: ...


@runtime_checkable
class FareEstimator(Protocol):
    model_version: str

    def estimate(self, req: FareRequest) -> FareEstimate: ...


@runtime_checkable
class Translator(Protocol):
    model_id: str

    async def translate(
        self, text: str, src: str, tgt: str, glossary: Sequence[GlossaryEntry]
    ) -> Translation: ...
