"""Offline fare estimator: a typical economy round trip, from parameters, no network.

    per traveller = (fixed + sum of concave per-km band rates over the one-way distance) x 2
                    x season[quarter of departure] x advance[days between booking and departure]

low and high are the middle times the residual quantiles of the parameter set: narrower when both
airports are in one country, wider across borders. v0 parameters are set by hand and say so; a
fitted set records its datasets and error metrics in the same file, and `fare_mdape` is null
until one exists.
"""

import bisect
import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from dentabridge_core import FareEstimate, FareRequest, Money, PriceRange
from dentabridge_pricing.airports import DATA_DIR, Airport, haversine_km, latest


class UnknownAirport(KeyError):
    pass


class FareParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    basis: str
    fixed_usd: float
    band_edges_km: tuple[float, ...]
    band_rate_usd_per_km: tuple[float, ...]
    season_by_quarter: tuple[float, float, float, float]
    advance_edges_days: tuple[int, ...]
    advance_factor: tuple[float, ...]
    residual_domestic: tuple[float, float]
    residual_international: tuple[float, float]
    fit: date | None
    dataset_versions: tuple[str, ...]
    fare_mdape: float | None
    interval_coverage: float | None
    bias_by_band: dict[str, float] | None


def load_fare_params(path: Path | None = None) -> FareParams:
    return FareParams.model_validate(
        json.loads((path or latest("fare_params_*.json", DATA_DIR)).read_text())
    )


class OfflineFareEstimator:
    def __init__(self, params: FareParams, airports: dict[str, Airport]) -> None:
        self._params = params
        self._airports = airports
        self.model_version = f"offline-{params.version}"

    @property
    def mdape(self) -> float | None:
        return self._params.fare_mdape

    def _airport(self, iata: str) -> Airport:
        try:
            return self._airports[iata]
        except KeyError:
            raise UnknownAirport(iata) from None

    def _one_way_usd(self, km: float) -> float:
        p = self._params
        edges = (*p.band_edges_km, float("inf"))
        total = p.fixed_usd
        for rate, low, high in zip(p.band_rate_usd_per_km, edges[:-1], edges[1:], strict=True):
            total += rate * max(0.0, min(km, high) - low)
        return total

    def estimate(self, req: FareRequest) -> FareEstimate:
        origin, dest = self._airport(req.origin_iata), self._airport(req.dest_iata)
        if origin.iata == dest.iata:
            raise ValueError("origin and destination are the same airport")
        days = (req.depart - req.booked_on).days
        if days < 0:
            raise ValueError("departure is before the booking date")
        p = self._params
        bin_ = bisect.bisect_right(p.advance_edges_days, days) - 1
        quarter = (req.depart.month - 1) // 3
        mid = (
            self._one_way_usd(haversine_km(origin, dest))
            * 2
            * p.season_by_quarter[quarter]
            * p.advance_factor[bin_]
            * req.travellers
        )
        international = origin.iso_country != dest.iso_country
        q_low, q_high = p.residual_international if international else p.residual_domestic

        def usd(amount: float) -> Money:
            return Money(amount_minor=round(amount * 100), currency="USD")

        return FareEstimate(
            range_usd=PriceRange(low=usd(mid * q_low), mid=usd(mid), high=usd(mid * q_high)),
            interval="q10-q90" if international else "q20-q80",
            international=international,
            basis_key=p.basis,
        )
