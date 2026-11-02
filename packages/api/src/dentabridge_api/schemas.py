"""Response bodies owned by the API itself (the rest live in dentabridge_core)."""

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import Field

from dentabridge_core import Contract, CountryCode, FindingCode, Iata, LanguageTag, Modality


class Health(Contract):
    status: Literal["ok"]
    contract_version: str
    mode: Literal["demo", "live"]
    vision_backend: str


class Meta(Contract):
    uploads_enabled: bool
    assessable: dict[Modality, list[FindingCode]]
    texts: dict[str, str]
    verified_pairs: list[tuple[str, str]]
    currencies: list[str] = Field(description="Only currencies the dated FX snapshot holds")
    fx_as_of: date | None


class SampleImage(Contract):
    id: str
    attribution: str
    licence: str


class AnalyseSampleRequest(Contract):
    sample_id: str
    locale: Literal["en", "ru"]


class City(Contract):
    iata: Iata
    city: str
    airport: str
    country_code: CountryCode


class HoldRequest(Contract):
    slot_id: UUID


class CallSessionRequest(Contract):
    patient_language: LanguageTag
    dentist_language: LanguageTag
