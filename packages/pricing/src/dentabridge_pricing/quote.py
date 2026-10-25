"""Quote assembly: treatment (SAMPLE), flights (ESTIMATE) and hotel (ESTIMATE), in one currency."""

import math
from datetime import datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from dentabridge_core import (
    ClinicSummary,
    FareEstimator,
    FareRequest,
    Money,
    PriceRange,
    Quote,
    QuoteLine,
    QuoteRequest,
)
from dentabridge_pricing.fx import FxSnapshot, convert
from dentabridge_pricing.hotels import HotelTable

VALID_FOR = timedelta(hours=24)


class TripTooShort(ValueError):
    pass


def _convert(r: PriceRange, to: str, fx: FxSnapshot) -> PriceRange:
    return PriceRange(
        low=convert(r.low, to, fx), mid=convert(r.mid, to, fx), high=convert(r.high, to, fx)
    )


def _scale(r: PriceRange, factor: int) -> PriceRange:
    def times(m: Money) -> Money:
        return Money(amount_minor=m.amount_minor * factor, currency=m.currency)

    return PriceRange(low=times(r.low), mid=times(r.mid), high=times(r.high))


def _sum(ranges: list[PriceRange], currency: str) -> PriceRange:
    def total(pick: str) -> Money:
        return Money(
            amount_minor=sum(getattr(r, pick).amount_minor for r in ranges), currency=currency
        )

    return PriceRange(low=total("low"), mid=total("mid"), high=total("high"))


def quote_id(req: QuoteRequest, as_of: datetime) -> UUID:
    """The same request on the same day is the same quote."""
    return uuid5(NAMESPACE_URL, f"dentabridge:quote:{as_of.date()}:{req.model_dump_json()}")


def build_quote(
    req: QuoteRequest,
    clinic: ClinicSummary,
    hotel_tier: str,
    fares: FareEstimator,
    hotels: HotelTable,
    fx: FxSnapshot,
    as_of: datetime,
    fare_mdape: float | None = None,
) -> Quote:
    priced = {t.treatment_code: t for t in clinic.treatments}
    chosen = [priced[code] for code in req.treatment_codes]  # KeyError: the clinic lacks it
    trips = max(t.trips for t in chosen)
    nights = max(t.nights_per_trip for t in chosen)
    if (req.return_date - req.depart_date).days < nights:
        raise TripTooShort(f"the treatment needs {nights} nights per trip")
    currency = req.currency

    treatment_usd = PriceRange(
        low=Money(amount_minor=sum(t.price.amount_minor for t in chosen), currency="USD"),
        mid=Money(amount_minor=sum(t.price.amount_minor for t in chosen), currency="USD"),
        high=Money(amount_minor=sum(t.price.amount_minor for t in chosen), currency="USD"),
    )
    fare = fares.estimate(
        FareRequest(
            origin_iata=req.origin_iata,
            dest_iata=clinic.airport_iata,
            depart=req.depart_date,
            return_=req.return_date,
            booked_on=as_of.date(),
            travellers=req.travellers,
        )
    )
    rooms = math.ceil(req.travellers / 2)
    hotel_usd = _scale(hotels.per_night_usd(hotel_tier), nights * trips * rooms)

    lines = (
        QuoteLine(
            kind="treatment",
            trips=trips,
            amount=_convert(treatment_usd, currency, fx),
            chip="SAMPLE",
            basis_key="clinic_sample_price",
        ),
        QuoteLine(
            kind="flight",
            trips=trips,
            amount=_convert(_scale(fare.range_usd, trips), currency, fx),
            chip="ESTIMATE",
            basis_key=fare.basis_key,
        ),
        QuoteLine(
            kind="hotel",
            trips=trips,
            amount=_convert(hotel_usd, currency, fx),
            chip="ESTIMATE",
            basis_key=f"hotel_tier_{hotel_tier}",
        ),
    )
    return Quote(
        quote_id=quote_id(req, as_of),
        request=req,
        lines=lines,
        total=_sum([line.amount for line in lines], currency),
        total_chip="ESTIMATE",
        fx_as_of=fx.as_of,
        fare_model_version=fares.model_version,
        fare_mdape=fare_mdape,
        valid_until=as_of + VALID_FOR,
    )
