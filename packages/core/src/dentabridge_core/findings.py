"""What the vision model may report, and the rules that keep one report coherent."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, model_validator

from dentabridge_core._base import Contract

Modality = Literal["intraoral_photo", "panoramic_xray"]
FindingStatus = Literal["possible", "not_seen", "not_assessed"]
Backend = Literal["ollama", "mlx", "transformers", "fixture"]


class FindingCode(StrEnum):
    CARIES = "caries"
    FILLING = "filling"
    CROWN = "crown"
    ABRASION = "abrasion"
    GINGIVITIS = "gingivitis"
    CALCULUS = "calculus"
    ULCER = "ulcer"
    DISCOLOURATION = "discolouration"
    HYPODONTIA = "hypodontia"
    MALOCCLUSION = "malocclusion"
    PERIAPICAL_LESION = "periapical_lesion"
    IMPACTED_TOOTH = "impacted_tooth"


# Codes a model may judge per modality. Everything else is reported as not_assessed.
ASSESSABLE: Mapping[Modality, frozenset[FindingCode]] = MappingProxyType(
    {
        "intraoral_photo": frozenset(
            {
                FindingCode.CARIES,
                FindingCode.FILLING,
                FindingCode.CROWN,
                FindingCode.ABRASION,
                FindingCode.GINGIVITIS,
            }
        ),
        "panoramic_xray": frozenset(
            {FindingCode.CARIES, FindingCode.PERIAPICAL_LESION, FindingCode.IMPACTED_TOOTH}
        ),
    }
)


class Region(Contract):
    """Box in image fractions with the origin at the top left."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("a region needs x0 < x1 and y0 < y1")
        return self


class Finding(Contract):
    code: FindingCode
    status: FindingStatus
    score: float | None = Field(default=None, ge=0, le=1)
    subtype: str | None = None  # source label, never shown to patients in v1
    regions: tuple[Region, ...] = ()

    @model_validator(mode="after")
    def _status_rules(self) -> Self:
        if self.status == "not_assessed" and self.score is not None:
            raise ValueError("a not_assessed finding has no score")
        if self.status != "possible" and self.regions:
            raise ValueError("only a possible finding has regions")
        return self


class ModelInfo(Contract):
    model_id: str
    version: str
    backend: Backend
    card_url: str | None = None
