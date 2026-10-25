"""Fictional demo clinics, their dentists and their sample treatment prices."""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from dentabridge_core._base import Contract, CountryCode, Iata, LanguageTag, Slug
from dentabridge_core.money import Money


class TreatmentCode(StrEnum):
    CONSULTATION = "consultation"
    FILLING = "filling"
    CROWN = "crown"
    ROOT_CANAL = "root_canal"
    IMPLANT = "implant"
    SCALING = "scaling"


class DentistSummary(Contract):
    dentist_id: Slug
    display_name: str
    languages: tuple[LanguageTag, ...] = Field(min_length=1)


class TreatmentPrice(Contract):
    treatment_code: TreatmentCode
    price: Money
    chip: Literal["SAMPLE"]
    trips: int = Field(ge=1)
    nights_per_trip: int = Field(ge=0)


class ClinicSummary(Contract):
    clinic_id: Slug
    name: str
    city: str
    country_code: CountryCode
    airport_iata: Iata
    timezone: str  # IANA name; the seed loader checks that it exists
    languages: tuple[LanguageTag, ...] = Field(min_length=1)
    dentists: tuple[DentistSummary, ...] = Field(min_length=1)
    treatments: tuple[TreatmentPrice, ...] = Field(min_length=1)
    is_fictional: Literal[True]
