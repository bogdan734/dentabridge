"""Money in minor units and the ranges every price is shown as."""

from typing import Literal, Self

from pydantic import Field, model_validator

from dentabridge_core._base import Contract, CurrencyCode

PriceChip = Literal["ESTIMATE", "SAMPLE", "SIMULATED"]


class Money(Contract):
    amount_minor: int = Field(ge=0)
    currency: CurrencyCode


class PriceRange(Contract):
    low: Money
    mid: Money
    high: Money

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not (self.low.currency == self.mid.currency == self.high.currency):
            raise ValueError("a price range has one currency")
        if not (self.low.amount_minor <= self.mid.amount_minor <= self.high.amount_minor):
            raise ValueError("a price range needs low <= mid <= high")
        return self
