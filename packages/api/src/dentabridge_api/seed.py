"""The fictional clinics of seed/clinics.yaml, validated."""

import itertools
import re
from datetime import time
from pathlib import Path
from typing import Literal, Self
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dentabridge_core import (
    ClinicSummary,
    CountryCode,
    DentistSummary,
    Iata,
    LanguageTag,
    Money,
    Slug,
    TreatmentCode,
    TreatmentPrice,
)

SEED_DIR = Path(__file__).resolve().parents[4] / "seed"
Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAYS: tuple[Weekday, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WINDOW = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Window(_Frozen):
    start: time
    end: time

    @classmethod
    def parse(cls, text: str) -> "Window":
        match = _WINDOW.match(text)
        if match is None:
            raise ValueError(f"not a HH:MM-HH:MM window: {text!r}")
        h1, m1, h2, m2 = (int(g) for g in match.groups())
        window = cls(start=time(h1, m1), end=time(h2, m2))
        if window.end <= window.start or window.start.minute or window.end.minute:
            raise ValueError(f"a window is whole hours with the end after the start: {text!r}")
        return window


class DentistSeed(_Frozen):
    dentist_id: Slug
    display_name: str
    languages: tuple[LanguageTag, ...] = Field(min_length=1)
    weekly_hours: dict[Weekday, tuple[Window, ...]]

    @field_validator("weekly_hours", mode="before")
    @classmethod
    def _windows(cls, value: dict[str, list[str]]) -> dict[str, tuple[Window, ...]]:
        return {day: tuple(Window.parse(w) for w in windows) for day, windows in value.items()}

    @model_validator(mode="after")
    def _no_overlap(self) -> Self:
        for windows in self.weekly_hours.values():
            ordered = sorted(windows, key=lambda w: w.start)
            if any(a.end > b.start for a, b in itertools.pairwise(ordered)):
                raise ValueError(f"{self.dentist_id}: overlapping windows")
        return self


class TreatmentSeed(_Frozen):
    treatment_code: TreatmentCode
    price_usd: int = Field(gt=0)
    trips: int = Field(ge=1)
    nights_per_trip: int = Field(ge=0)


class ClinicSeed(_Frozen):
    clinic_id: Slug
    name: str
    city: str
    country_code: CountryCode
    airport_iata: Iata
    timezone: str
    languages: tuple[LanguageTag, ...] = Field(min_length=1)
    hotel_tier: str
    dentists: tuple[DentistSeed, ...] = Field(min_length=1)
    treatments: tuple[TreatmentSeed, ...] = Field(min_length=1)
    is_fictional: Literal[True]

    @field_validator("timezone")
    @classmethod
    def _zone_exists(cls, value: str) -> str:
        ZoneInfo(value)  # raises ZoneInfoNotFoundError, which pydantic reports as a value error
        return value

    def summary(self) -> ClinicSummary:
        return ClinicSummary(
            clinic_id=self.clinic_id,
            name=self.name,
            city=self.city,
            country_code=self.country_code,
            airport_iata=self.airport_iata,
            timezone=self.timezone,
            languages=self.languages,
            dentists=tuple(
                DentistSummary(
                    dentist_id=d.dentist_id, display_name=d.display_name, languages=d.languages
                )
                for d in self.dentists
            ),
            treatments=tuple(
                TreatmentPrice(
                    treatment_code=t.treatment_code,
                    price=Money(amount_minor=t.price_usd * 100, currency="USD"),
                    chip="SAMPLE",
                    trips=t.trips,
                    nights_per_trip=t.nights_per_trip,
                )
                for t in self.treatments
            ),
            is_fictional=True,
        )


def load_clinics(path: Path | None = None) -> dict[str, ClinicSeed]:
    raw = yaml.safe_load((path or SEED_DIR / "clinics.yaml").read_text(encoding="utf-8"))
    clinics = [ClinicSeed.model_validate(item) for item in raw]
    ids = [c.clinic_id for c in clinics]
    if len(ids) != len(set(ids)):
        raise ValueError("clinic ids must be unique")
    return {c.clinic_id: c for c in clinics}
