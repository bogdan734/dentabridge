"""A trip quote: treatment, flights and hotel, every figure a range with a chip."""

from datetime import date
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from dentabridge_core._base import Contract, CurrencyCode, Iata, Slug
from dentabridge_core.clinic import TreatmentCode
from dentabridge_core.money import PriceChip, PriceRange


class QuoteRequest(Contract):
    clinic_id: Slug
    treatment_codes: tuple[TreatmentCode, ...] = Field(min_length=1)
    origin_iata: Iata
    depart_date: date
    return_date: date
    travellers: int = Field(ge=1, le=6)
    currency: CurrencyCode

    @model_validator(mode="after")
    def _return_not_before_depart(self) -> Self:
        # A trip shorter than the treatment needs is the API's trip_too_short, not a schema error.
        if self.return_date < self.depart_date:
            raise ValueError("return_date is before depart_date")
        return self


class QuoteLine(Contract):
    kind: Literal["treatment", "flight", "hotel"]
    trips: int = Field(ge=1)
    amount: PriceRange
    chip: PriceChip
    basis_key: str


class Quote(Contract):
    quote_id: UUID  # uuid5 of the canonical request plus the as-of date
    request: QuoteRequest
    lines: tuple[QuoteLine, ...] = Field(min_length=1)
    total: PriceRange
    total_chip: Literal["ESTIMATE"]
    fx_as_of: date
    fare_model_version: str
    fare_mdape: float | None = Field(default=None, ge=0)
    valid_until: AwareDatetime

    @model_validator(mode="after")
    def _one_currency(self) -> Self:
        currencies = {line.amount.low.currency for line in self.lines}
        currencies.add(self.total.low.currency)
        if currencies != {self.request.currency}:
            raise ValueError("every line and the total use the requested currency")
        return self
