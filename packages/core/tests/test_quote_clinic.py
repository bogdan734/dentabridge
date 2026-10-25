from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from dentabridge_core import ClinicSummary, Money, PriceRange, Quote, QuoteRequest

REQUEST: dict[str, object] = {
    "clinic_id": "demo-clinic",
    "treatment_codes": ["crown"],
    "origin_iata": "LHR",
    "depart_date": date(2026, 12, 1),
    "return_date": date(2026, 12, 8),
    "travellers": 1,
    "currency": "EUR",
}


def _range(currency: str) -> dict[str, object]:
    def money(cents: int) -> dict[str, object]:
        return {"amount_minor": cents, "currency": currency}

    return {"low": money(100), "mid": money(200), "high": money(300)}


def _quote(line_currency: str = "EUR", total_currency: str = "EUR") -> Quote:
    return Quote.model_validate(
        {
            "quote_id": "00000000-0000-0000-0000-000000000001",
            "request": REQUEST,
            "lines": [
                {
                    "kind": "flight",
                    "trips": 1,
                    "amount": _range(line_currency),
                    "chip": "ESTIMATE",
                    "basis_key": "fare.v0",
                }
            ],
            "total": _range(total_currency),
            "total_chip": "ESTIMATE",
            "fx_as_of": date(2026, 10, 30),
            "fare_model_version": "v0",
            "valid_until": datetime(2026, 11, 3, tzinfo=UTC),
        }
    )


@pytest.mark.parametrize(
    "change",
    [
        {"travellers": 0},
        {"travellers": 7},
        {"treatment_codes": []},
        {"origin_iata": "lhr"},
        {"return_date": date(2026, 11, 30)},
    ],
)
def test_quote_request_bounds(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        QuoteRequest.model_validate({**REQUEST, **change})


def test_quote_uses_one_currency() -> None:
    assert _quote().fare_mdape is None
    with pytest.raises(ValidationError, match="requested currency"):
        _quote(line_currency="USD")
    with pytest.raises(ValidationError, match="requested currency"):
        _quote(total_currency="USD")


def test_quote_total_chip_is_estimate() -> None:
    data = _quote().model_dump()
    with pytest.raises(ValidationError):
        Quote.model_validate({**data, "total_chip": "SAMPLE"})


def test_clinic_must_be_fictional() -> None:
    clinic: dict[str, object] = {
        "clinic_id": "demo-clinic",
        "name": "Demo Dental",
        "city": "Anytown",
        "country_code": "XX",
        "airport_iata": "AAA",
        "timezone": "Europe/London",
        "languages": ["en"],
        "dentists": [{"dentist_id": "dr-demo", "display_name": "Dr Demo", "languages": ["en"]}],
        "treatments": [
            {
                "treatment_code": "crown",
                "price": Money(amount_minor=50000, currency="EUR").model_dump(),
                "chip": "SAMPLE",
                "trips": 2,
                "nights_per_trip": 3,
            }
        ],
        "is_fictional": True,
    }
    assert ClinicSummary.model_validate(clinic).treatments[0].trips == 2
    with pytest.raises(ValidationError):
        ClinicSummary.model_validate({**clinic, "is_fictional": False})
    with pytest.raises(ValidationError):
        ClinicSummary.model_validate({**clinic, "dentists": []})
    assert PriceRange.model_validate(_range("EUR")).mid.amount_minor == 200
