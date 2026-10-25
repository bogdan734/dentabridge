"""Hand-set hotel ranges per room and night, by city tier (seed/hotel_tiers.yaml)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from dentabridge_core import Money, PriceRange

SEED_DIR = Path(__file__).resolve().parents[4] / "seed"


class _Tier(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    low: int
    mid: int
    high: int


class HotelTable:
    def __init__(self, tiers: dict[str, _Tier]) -> None:
        self._tiers = tiers

    def per_night_usd(self, tier: str) -> PriceRange:
        t = self._tiers[tier]

        def usd(amount: int) -> Money:
            return Money(amount_minor=amount * 100, currency="USD")

        return PriceRange(low=usd(t.low), mid=usd(t.mid), high=usd(t.high))

    def tiers(self) -> list[str]:
        return sorted(self._tiers)


def load_hotels(path: Path | None = None) -> HotelTable:
    raw = yaml.safe_load((path or SEED_DIR / "hotel_tiers.yaml").read_text(encoding="utf-8"))
    return HotelTable({name: _Tier.model_validate(t) for name, t in raw["tiers"].items()})
