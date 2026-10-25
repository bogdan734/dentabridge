"""A dated snapshot of ECB reference rates; a currency it lacks is refused, never guessed."""

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from dentabridge_core import Money
from dentabridge_pricing.airports import latest

# ISO 4217 minor units; everything else the ECB publishes has two decimals.
ZERO_DECIMAL = frozenset({"JPY", "KRW", "ISK"})


class CurrencyUnsupported(ValueError):
    pass


class FxSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: date
    base: str  # the currency every rate is quoted against
    rates: dict[str, float]  # units of the currency per one unit of base

    def currencies(self) -> list[str]:
        return sorted({self.base, *self.rates})

    def rate(self, currency: str) -> float:
        if currency == self.base:
            return 1.0
        try:
            return self.rates[currency]
        except KeyError:
            raise CurrencyUnsupported(currency) from None


def load_fx(path: Path | None = None) -> FxSnapshot:
    return FxSnapshot.model_validate(json.loads((path or latest("fx_*.json")).read_text()))


def _factor(currency: str) -> int:
    return 1 if currency in ZERO_DECIMAL else 100


def convert(money: Money, to: str, fx: FxSnapshot) -> Money:
    """Through the base currency; the result is rounded to the target's minor unit."""
    major = money.amount_minor / _factor(money.currency) / fx.rate(money.currency)
    return Money(amount_minor=round(major * fx.rate(to) * _factor(to)), currency=to)
