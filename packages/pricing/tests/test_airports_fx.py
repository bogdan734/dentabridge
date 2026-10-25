import pytest

from dentabridge_core import Money
from dentabridge_pricing.airports import haversine_km, load_airports, search_cities
from dentabridge_pricing.fx import CurrencyUnsupported, FxSnapshot, convert, load_fx

AIRPORTS = load_airports()
FX = load_fx()


def test_haversine_lhr_jfk() -> None:
    assert haversine_km(AIRPORTS["LHR"], AIRPORTS["JFK"]) == pytest.approx(5540, abs=10)


def test_haversine_is_symmetric_and_zero_to_itself() -> None:
    a, b = AIRPORTS["LIS"], AIRPORTS["BKK"]
    assert haversine_km(a, b) == pytest.approx(haversine_km(b, a))
    assert haversine_km(a, a) == 0


def test_search_case_insensitive_prefix() -> None:
    lower = search_cities("lond", 20, AIRPORTS)
    assert {a.iata for a in lower} >= {"LHR", "LGW"}
    assert lower == search_cities("LOND", 20, AIRPORTS)
    assert len(search_cities("a", 5, AIRPORTS)) == 5
    assert search_cities("   ", 5, AIRPORTS) == []
    assert search_cities("zzzzqq", 5, AIRPORTS) == []
    assert next(a.iata for a in search_cities("hkg", 3, AIRPORTS)) == "HKG"


def test_only_scheduled_iata_airports() -> None:
    assert len(AIRPORTS) > 3000
    assert all(len(code) == 3 and code.isupper() for code in AIRPORTS)
    assert all(-90 <= a.lat <= 90 and -180 <= a.lon <= 180 for a in AIRPORTS.values())
    assert {"LHR", "JFK", "LIS", "IST", "BKK", "SYD"} <= set(AIRPORTS)


def test_convert_roundtrip_within_one_minor() -> None:
    start = Money(amount_minor=123_456, currency="USD")
    for currency in ("EUR", "GBP", "JPY", "THB"):
        there = convert(start, currency, FX)
        assert there.currency == currency
        assert abs(convert(there, "USD", FX).amount_minor - start.amount_minor) <= max(
            1, round(100 / FX.rate(currency))
        )


def test_zero_decimal_currencies_use_whole_units() -> None:
    hundred_dollars = convert(Money(amount_minor=10_000, currency="USD"), "JPY", FX)
    assert hundred_dollars.amount_minor == round(100 * FX.rate("JPY"))


def test_unsupported_currency_raises_and_rub_is_not_offered() -> None:
    with pytest.raises(CurrencyUnsupported):
        convert(Money(amount_minor=100, currency="USD"), "RUB", FX)
    with pytest.raises(CurrencyUnsupported):
        convert(Money(amount_minor=100, currency="RUB"), "USD", FX)
    assert "RUB" not in FX.currencies()
    assert "USD" in FX.currencies()


def test_snapshot_is_dated_and_closed() -> None:
    assert FX.as_of.year >= 2026
    with pytest.raises(ValueError, match="extra"):
        FxSnapshot(as_of=FX.as_of, base="USD", rates={}, extra=1)  # type: ignore[call-arg]
