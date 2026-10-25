from datetime import UTC, date, datetime, timedelta

import pytest

from dentabridge_core import (
    ClinicSummary,
    DentistSummary,
    Money,
    Quote,
    QuoteLine,
    QuoteRequest,
    TreatmentCode,
    TreatmentPrice,
)
from dentabridge_core.testing import FakeFareEstimator
from dentabridge_pricing.fx import CurrencyUnsupported, load_fx
from dentabridge_pricing.hotels import load_hotels
from dentabridge_pricing.quote import TripTooShort, build_quote, quote_id

FX = load_fx()
HOTELS = load_hotels()
AS_OF = datetime(2026, 10, 25, 12, tzinfo=UTC)


def _price(code: TreatmentCode, usd: int, trips: int, nights: int) -> TreatmentPrice:
    return TreatmentPrice(
        treatment_code=code,
        price=Money(amount_minor=usd * 100, currency="USD"),
        chip="SAMPLE",
        trips=trips,
        nights_per_trip=nights,
    )


CLINIC = ClinicSummary(
    clinic_id="demo-clinic",
    name="Demo Dental",
    city="Lisbon",
    country_code="PT",
    airport_iata="LIS",
    timezone="Europe/Lisbon",
    languages=("en",),
    dentists=(DentistSummary(dentist_id="dr-demo", display_name="Dr. Demo", languages=("en",)),),
    treatments=(
        _price(TreatmentCode.CONSULTATION, 50, 1, 1),
        _price(TreatmentCode.CROWN, 400, 1, 5),
        _price(TreatmentCode.IMPLANT, 1500, 2, 4),
    ),
    is_fictional=True,
)


def request(
    codes: tuple[TreatmentCode, ...], nights: int = 7, travellers: int = 1, currency: str = "USD"
) -> QuoteRequest:
    depart = date(2026, 12, 1)
    return QuoteRequest(
        clinic_id="demo-clinic",
        treatment_codes=codes,
        origin_iata="LHR",
        depart_date=depart,
        return_date=depart + timedelta(days=nights),
        travellers=travellers,
        currency=currency,
    )


def quote(req: QuoteRequest) -> Quote:
    return build_quote(req, CLINIC, "standard", FakeFareEstimator(), HOTELS, FX, AS_OF)


def line(q: Quote, kind: str) -> QuoteLine:
    return next(item for item in q.lines if item.kind == kind)


def test_total_bounds_are_sums() -> None:
    q = quote(request((TreatmentCode.CROWN,)))
    for bound in ("low", "mid", "high"):
        assert getattr(q.total, bound).amount_minor == sum(
            getattr(item.amount, bound).amount_minor for item in q.lines
        )
    assert q.total_chip == "ESTIMATE"


def test_every_line_has_chip_and_basis() -> None:
    q = quote(request((TreatmentCode.CROWN,)))
    assert [(item.kind, item.chip) for item in q.lines] == [
        ("treatment", "SAMPLE"),
        ("flight", "ESTIMATE"),
        ("hotel", "ESTIMATE"),
    ]
    assert all(item.basis_key for item in q.lines)


def test_two_trip_treatment_doubles_flights() -> None:
    one = quote(request((TreatmentCode.CROWN,)))
    two = quote(request((TreatmentCode.IMPLANT,)))
    assert line(one, "flight").trips == 1
    assert line(two, "flight").trips == 2
    assert (
        line(two, "flight").amount.mid.amount_minor
        == 2 * line(one, "flight").amount.mid.amount_minor
    )
    # implant: 4 nights x 2 trips = 8 room-nights at 90 USD; crown: 5 nights x 1 trip
    assert line(two, "hotel").amount.mid.amount_minor == 8 * 9000
    assert line(one, "hotel").amount.mid.amount_minor == 5 * 9000


def test_rooms_follow_travellers_in_pairs() -> None:
    three = quote(request((TreatmentCode.CROWN,), travellers=3))
    assert line(three, "hotel").amount.mid.amount_minor == 5 * 2 * 9000  # ceil(3 / 2) = 2 rooms


def test_treatments_are_summed_and_trips_and_nights_are_the_maximum() -> None:
    q = quote(request((TreatmentCode.CONSULTATION, TreatmentCode.IMPLANT)))
    treatment = q.lines[0]
    assert treatment.amount.mid.amount_minor == (50 + 1500) * 100
    assert q.lines[1].trips == 2


def test_quote_id_stable() -> None:
    req = request((TreatmentCode.CROWN,))
    assert quote(req).quote_id == quote(req).quote_id == quote_id(req, AS_OF)
    assert quote_id(req, AS_OF + timedelta(days=1)) != quote_id(req, AS_OF)
    assert quote_id(request((TreatmentCode.CROWN,), nights=8), AS_OF) != quote_id(req, AS_OF)


def test_trip_too_short() -> None:
    with pytest.raises(TripTooShort):
        quote(request((TreatmentCode.CROWN,), nights=4))


def test_currency_conversion_and_unsupported_currency() -> None:
    euro = quote(request((TreatmentCode.CROWN,), currency="EUR"))
    assert {line.amount.mid.currency for line in euro.lines} == {"EUR"}
    assert euro.fx_as_of == FX.as_of
    with pytest.raises(CurrencyUnsupported):
        quote(request((TreatmentCode.CROWN,), currency="RUB"))


def test_valid_for_a_day_and_records_the_fare_model() -> None:
    q = quote(request((TreatmentCode.CROWN,)))
    assert q.valid_until == AS_OF + timedelta(hours=24)
    assert q.fare_model_version == "fake-1"
    assert q.fare_mdape is None
