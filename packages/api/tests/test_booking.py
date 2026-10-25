from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest

from dentabridge_api.booking.slots import generate_slots
from dentabridge_api.clock import FrozenClock
from dentabridge_api.seed import ClinicSeed, DentistSeed, TreatmentSeed, Window, load_clinics
from dentabridge_core import ErrorCode, ErrorResponse, TreatmentCode

CLINIC = "alfama-demo-dental"
_keys = iter(range(1, 10_000))


def key() -> dict[str, str]:
    return {"Idempotency-Key": f"booking-{next(_keys):05d}"}


def code_of(response: httpx.Response) -> ErrorCode:
    return ErrorResponse.model_validate(response.json()).code


def quote(client: Any, **changes: object) -> dict[str, Any]:
    body = {
        "clinic_id": CLINIC,
        "treatment_codes": ["crown"],
        "origin_iata": "LHR",
        "depart_date": "2026-11-10",
        "return_date": "2026-11-20",
        "travellers": 1,
        "currency": "EUR",
        **changes,
    }
    response = client.post("/v1/quotes", json=body)
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


def first_slot(
    client: Any, clinic: str = CLINIC, start: str = "2026-11-11", end: str = "2026-11-14"
) -> dict[str, Any]:
    slots = client.get(f"/v1/clinics/{clinic}/slots", params={"from": start, "to": end}).json()
    assert slots, "the seed schedule has slots in the window"
    result: dict[str, Any] = slots[0]
    return result


def hold(client: Any, slot_id: str) -> httpx.Response:
    response: httpx.Response = client.post("/v1/holds", json={"slot_id": slot_id}, headers=key())
    return response


def book(client: Any, hold_id: str, quote_id: str, **extra: str) -> httpx.Response:
    response: httpx.Response = client.post(
        "/v1/bookings", json={"hold_id": hold_id, "quote_id": quote_id, **extra}, headers=key()
    )
    return response


def test_seed_all_fictional(make_client: Any) -> None:
    clinics = load_clinics()
    assert len(clinics) == 4
    assert all(c.is_fictional for c in clinics.values())
    assert all("Demo" in c.name for c in clinics.values()), "no real-sounding clinic names"
    continents = {c.timezone.split("/")[0] for c in clinics.values()}
    assert len(continents) >= 3
    january, july = datetime(2027, 1, 15, tzinfo=UTC), datetime(2027, 7, 15, tzinfo=UTC)
    shifts = [
        ZoneInfo(c.timezone).utcoffset(january) != ZoneInfo(c.timezone).utcoffset(july)
        for c in clinics.values()
    ]
    assert any(shifts), "one clinic observes daylight saving time"
    assert not all(shifts), "and one does not"
    listed = make_client().get("/v1/clinics").json()
    assert {c["clinic_id"] for c in listed} == set(clinics)
    assert all(c["is_fictional"] is True for c in listed)


def test_no_country_city_or_currency_literals_in_code() -> None:
    from pathlib import Path

    roots = [
        Path(__file__).resolve().parents[2] / "api" / "src",
        Path(__file__).resolve().parents[2] / "pricing" / "src",
    ]
    names = {n for c in load_clinics().values() for n in (c.city, c.name)}
    text = "".join(p.read_text() for root in roots for p in root.rglob("*.py"))
    assert not [n for n in names if n in text]


def window_clinic(timezone: str, hours: str) -> ClinicSeed:
    return ClinicSeed(
        clinic_id="test-clinic",
        name="Test Demo",
        city="Nowhere",
        country_code="PT",
        airport_iata="LIS",
        timezone=timezone,
        languages=("en",),
        hotel_tier="standard",
        dentists=(
            DentistSeed(
                dentist_id="dr-test",
                display_name="Dr. Test",
                languages=("en",),
                weekly_hours={day: (Window.parse(hours),) for day in ("sat", "sun", "mon")},
            ),
        ),
        treatments=(
            TreatmentSeed(
                treatment_code=TreatmentCode.CONSULTATION, price_usd=10, trips=1, nights_per_trip=1
            ),
        ),
        is_fictional=True,
    )


def test_dst_spring_forward_slots() -> None:
    # Lisbon clocks jump 01:00 -> 02:00 on Sunday 2027-03-28: local 01:00 does not exist
    clinic = window_clinic("Europe/Lisbon", "00:00-04:00")
    slots = generate_slots(clinic, datetime(2027, 3, 28).date(), 1)
    local = [s.start.astimezone(ZoneInfo("Europe/Lisbon")).strftime("%H:%M") for s in slots]
    assert local == ["00:00", "02:00", "03:00"]
    assert all(s.end - s.start == timedelta(minutes=60) for s in slots)


def test_dst_keeps_the_local_hour_and_moves_the_utc_instant() -> None:
    clinic = window_clinic("Europe/Lisbon", "09:00-10:00")
    before = generate_slots(clinic, datetime(2027, 3, 27).date(), 1)[0]  # Saturday, winter time
    after = generate_slots(clinic, datetime(2027, 3, 29).date(), 1)[0]  # Monday, summer time
    assert before.start.hour == 9
    assert after.start.hour == 8


def test_fall_back_hour_is_offered_once() -> None:
    clinic = window_clinic("Europe/Lisbon", "00:00-03:00")
    slots = generate_slots(clinic, datetime(2027, 10, 31).date(), 1)
    assert [s.start.astimezone(ZoneInfo("Europe/Lisbon")).hour for s in slots] == [0, 1, 2]


def test_slot_ids_are_stable() -> None:
    clinic = load_clinics()[CLINIC]
    first = generate_slots(clinic, datetime(2026, 11, 11).date(), 3)
    again = generate_slots(clinic, datetime(2026, 11, 11).date(), 3)
    assert [s.slot_id for s in first] == [s.slot_id for s in again]
    assert len({s.slot_id for s in first}) == len(first)


def test_lead_time_48h_and_the_90_day_horizon(make_client: Any, clock: FrozenClock) -> None:
    client = make_client()
    soonest = client.get(
        f"/v1/clinics/{CLINIC}/slots", params={"from": "2026-10-25", "to": "2026-11-04"}
    ).json()
    assert soonest, "slots exist"
    starts = [datetime.fromisoformat(s["start"]) for s in soonest]
    assert min(starts) >= clock.now() + timedelta(hours=48)
    beyond = client.get(
        f"/v1/clinics/{CLINIC}/slots", params={"from": "2027-02-10", "to": "2027-03-10"}
    ).json()
    assert max(datetime.fromisoformat(s["start"]) for s in beyond) <= clock.now() + timedelta(
        days=90
    )


def test_slot_listing_rules(make_client: Any) -> None:
    client = make_client()
    too_long = client.get(
        f"/v1/clinics/{CLINIC}/slots", params={"from": "2026-11-01", "to": "2026-12-15"}
    )
    assert (too_long.status_code, code_of(too_long)) == (422, ErrorCode.VALIDATION_ERROR)
    assert (
        client.get(
            "/v1/clinics/nowhere/slots", params={"from": "2026-11-01", "to": "2026-11-02"}
        ).status_code
        == 404
    )
    assert client.get("/v1/clinics/nowhere").status_code == 404
    only_crowns = client.get("/v1/clinics", params={"treatment": "implant"}).json()
    assert len(only_crowns) == 4


def test_a_held_slot_shows_as_held_and_a_booked_one_as_booked(make_client: Any) -> None:
    client = make_client()
    q = quote(client)
    slot = first_slot(client)
    held = hold(client, slot["slot_id"]).json()
    assert first_slot(client)["status"] == "held"
    assert book(client, held["hold_id"], q["quote_id"]).status_code == 201
    assert first_slot(client)["status"] == "booked"


def test_hold_expires_after_10_min(make_client: Any, clock: FrozenClock) -> None:
    client = make_client()
    q = quote(client)
    slot = first_slot(client)
    first = hold(client, slot["slot_id"])
    assert first.status_code == 201
    second = hold(client, slot["slot_id"])
    assert (second.status_code, code_of(second)) == (409, ErrorCode.SLOT_UNAVAILABLE)
    clock.advance(timedelta(minutes=10, seconds=1))
    late = book(client, first.json()["hold_id"], q["quote_id"])
    assert (late.status_code, code_of(late)) == (409, ErrorCode.HOLD_EXPIRED)
    assert hold(client, slot["slot_id"]).status_code == 201, "the slot is free again"


def test_released_hold_frees_the_slot(make_client: Any) -> None:
    client = make_client()
    slot = first_slot(client)
    held = hold(client, slot["slot_id"]).json()
    assert client.delete(f"/v1/holds/{held['hold_id']}").status_code == 204
    assert client.delete(f"/v1/holds/{held['hold_id']}").status_code == 204, (
        "releasing twice is fine"
    )
    assert hold(client, slot["slot_id"]).status_code == 201
    assert client.delete("/v1/holds/00000000-0000-4000-8000-000000000000").status_code == 404


def test_unknown_slot_404(make_client: Any) -> None:
    response = hold(make_client(), "00000000-0000-4000-8000-000000000000")
    assert (response.status_code, code_of(response)) == (404, ErrorCode.NOT_FOUND)


def test_booking_returns_simulated_tickets(make_client: Any) -> None:
    client = make_client()
    q = quote(client)
    held = hold(client, first_slot(client)["slot_id"]).json()
    booking = book(client, held["hold_id"], q["quote_id"]).json()
    assert booking["simulated"] is True
    assert booking["status"] == "confirmed"
    assert {t["kind"] for t in booking["tickets"]} == {"flight", "hotel"}
    assert all(
        t["chip"] == "SIMULATED" and t["reference"].startswith("DEMO-") for t in booking["tickets"]
    )
    assert client.get(f"/v1/bookings/{booking['booking_id']}").json() == booking
    assert client.get("/v1/bookings/00000000-0000-4000-8000-000000000000").status_code == 404


def test_quote_other_clinic_409(make_client: Any) -> None:
    client = make_client()
    other = quote(client, clinic_id="galata-demo-dental")
    held = hold(client, first_slot(client)["slot_id"]).json()
    response = book(client, held["hold_id"], other["quote_id"])
    assert (response.status_code, code_of(response)) == (409, ErrorCode.QUOTE_MISMATCH)


def test_outside_trip_409(make_client: Any) -> None:
    client = make_client()
    short = quote(client, depart_date="2026-12-01", return_date="2026-12-08")  # the slot is in Nov
    held = hold(client, first_slot(client)["slot_id"]).json()
    response = book(client, held["hold_id"], short["quote_id"])
    assert (response.status_code, code_of(response)) == (409, ErrorCode.OUTSIDE_TRIP)


def test_expired_quote_410(make_client: Any, clock: FrozenClock) -> None:
    client = make_client()
    q = quote(client)
    held = hold(client, first_slot(client)["slot_id"]).json()
    clock.advance(timedelta(minutes=5))
    assert client.get(f"/v1/quotes/{q['quote_id']}").status_code == 200
    # a day later the quote is gone, but the hold lapsed first; take a fresh hold on a later slot
    clock.advance(timedelta(hours=24))
    fresh = hold(client, first_slot(client, start="2026-11-12", end="2026-11-14")["slot_id"]).json()
    response = book(client, fresh["hold_id"], q["quote_id"])
    assert (response.status_code, code_of(response)) == (410, ErrorCode.QUOTE_EXPIRED)
    assert client.get(f"/v1/quotes/{q['quote_id']}").status_code == 410
    assert held


def test_cancel_idempotent(make_client: Any) -> None:
    client = make_client()
    q = quote(client)
    slot = first_slot(client)
    booking = book(client, hold(client, slot["slot_id"]).json()["hold_id"], q["quote_id"]).json()
    first = client.post(f"/v1/bookings/{booking['booking_id']}/cancel")
    again = client.post(f"/v1/bookings/{booking['booking_id']}/cancel")
    assert (first.status_code, again.status_code) == (200, 200)
    assert first.json() == again.json()
    assert first.json()["status"] == "cancelled"
    assert first_slot(client)["status"] == "available", "cancel frees the slot"
    assert hold(client, slot["slot_id"]).status_code == 201
    assert (
        client.post("/v1/bookings/00000000-0000-4000-8000-000000000000/cancel").status_code == 404
    )


def test_the_same_key_books_once(make_client: Any) -> None:
    client = make_client()
    q = quote(client)
    held = hold(client, first_slot(client)["slot_id"]).json()
    headers = {"Idempotency-Key": "booking-same-key"}
    body = {"hold_id": held["hold_id"], "quote_id": q["quote_id"]}
    first = client.post("/v1/bookings", json=body, headers=headers)
    again = client.post("/v1/bookings", json=body, headers=headers)
    assert again.headers["idempotent-replayed"] == "true"
    assert again.json() == first.json()


def consent(client: Any, purpose: str) -> str:
    response = client.post(
        "/v1/consents",
        json={
            "purpose": purpose,
            "health_data_explicit": True,
            "text_version": "v1",
            "locale": "en",
        },
    )
    return str(response.json()["consent_id"])


def an_analysis(client: Any) -> str:
    response = client.post(
        "/v1/analyses/sample", json={"sample_id": "one", "locale": "en"}, headers=key()
    )
    return str(response.json()["analysis_id"])


def test_share_requires_own_consent_403(make_client: Any) -> None:
    client = make_client()
    q = quote(client)
    analysis = an_analysis(client)
    wrong_purpose = consent(client, "photo_analysis")
    held = hold(client, first_slot(client)["slot_id"]).json()
    response = book(
        client, held["hold_id"], q["quote_id"], analysis_id=analysis, share_consent_id=wrong_purpose
    )
    assert (response.status_code, code_of(response)) == (403, ErrorCode.SHARE_CONSENT_REQUIRED)
    unknown = book(
        client,
        held["hold_id"],
        q["quote_id"],
        analysis_id=analysis,
        share_consent_id="00000000-0000-4000-8000-000000000000",
    )
    assert unknown.status_code == 403
    alone = client.post(
        "/v1/bookings",
        json={"hold_id": held["hold_id"], "quote_id": q["quote_id"], "analysis_id": analysis},
        headers=key(),
    )
    assert alone.status_code == 422, "an analysis cannot be shared without its consent"


def test_sharing_copies_the_snapshot_and_it_goes_on_cancel(make_client: Any) -> None:
    from sqlalchemy.orm import Session

    from dentabridge_api.db.models import BookingRow

    client = make_client()
    q = quote(client)
    analysis, share = an_analysis(client), consent(client, "share_with_clinic")
    held = hold(client, first_slot(client)["slot_id"]).json()
    booking = book(
        client, held["hold_id"], q["quote_id"], analysis_id=analysis, share_consent_id=share
    ).json()
    assert booking["shared_analysis_id"] == analysis
    sessions = client.app.state.sessions
    with sessions() as session:
        row = session.get(BookingRow, booking["booking_id"])
        assert row is not None and row.snapshot_json is not None
        assert "image" not in row.snapshot_json.lower().replace("image_quality", "")
    client.post(f"/v1/bookings/{booking['booking_id']}/cancel")
    with sessions() as session:
        assert isinstance(session, Session)
        assert session.get(BookingRow, booking["booking_id"]).snapshot_json is None  # type: ignore[union-attr]


def test_snapshot_purged_after_appointment_plus_24h(make_client: Any, clock: FrozenClock) -> None:
    from dentabridge_api.db.models import BookingRow

    client = make_client()
    q = quote(client)
    analysis, share = an_analysis(client), consent(client, "share_with_clinic")
    slot = first_slot(client)
    held = hold(client, slot["slot_id"]).json()
    book(client, held["hold_id"], q["quote_id"], analysis_id=analysis, share_consent_id=share)
    appointment_end = datetime.fromisoformat(slot["end"])
    clock.set(appointment_end + timedelta(hours=23))
    # any booking call runs the purge; a second clinic keeps this one from needing the first
    other = quote(
        client, clinic_id="galata-demo-dental", depart_date="2026-11-10", return_date="2026-11-20"
    )
    held2 = hold(client, first_slot(client, "galata-demo-dental")["slot_id"]).json()
    book(client, held2["hold_id"], other["quote_id"])
    with client.app.state.sessions() as session:
        assert session.query(BookingRow).filter(BookingRow.snapshot_json.is_not(None)).count() == 1
    clock.set(appointment_end + timedelta(hours=25))
    held3 = hold(
        client, first_slot(client, "galata-demo-dental", "2026-11-13", "2026-11-14")["slot_id"]
    ).json()
    book(client, held3["hold_id"], other["quote_id"])
    with client.app.state.sessions() as session:
        assert session.query(BookingRow).filter(BookingRow.snapshot_json.is_not(None)).count() == 0


@pytest.mark.parametrize(
    ("changes", "status", "code"),
    [
        ({"clinic_id": "nowhere"}, 404, ErrorCode.NOT_FOUND),
        ({"return_date": "2026-11-12"}, 422, ErrorCode.TRIP_TOO_SHORT),
        ({"currency": "RUB"}, 422, ErrorCode.CURRENCY_UNSUPPORTED),
        ({"origin_iata": "QQQ"}, 422, ErrorCode.UNKNOWN_AIRPORT),
        (
            {
                "treatment_codes": ["consultation", "scaling", "implant", "crown"],
                "origin_iata": "LIS",
            },
            422,
            ErrorCode.VALIDATION_ERROR,
        ),
    ],
)
def test_quote_errors(
    make_client: Any, changes: dict[str, Any], status: int, code: ErrorCode
) -> None:
    body = {
        "clinic_id": CLINIC,
        "treatment_codes": ["crown"],
        "origin_iata": "LHR",
        "depart_date": "2026-11-10",
        "return_date": "2026-11-20",
        "travellers": 1,
        "currency": "EUR",
        **changes,
    }
    response = make_client().post("/v1/quotes", json=body)
    assert (response.status_code, code_of(response)) == (status, code)


def test_quote_is_stable_for_the_same_request_and_day(make_client: Any, clock: FrozenClock) -> None:
    client = make_client()
    first, again = quote(client), quote(client)
    assert first["quote_id"] == again["quote_id"]
    clock.advance(timedelta(days=1))
    assert quote(client)["quote_id"] != first["quote_id"]


def test_cities_and_treatments(make_client: Any) -> None:
    client = make_client()
    assert "crown" in client.get("/v1/treatments").json()
    cities = client.get("/v1/cities", params={"q": "lisb", "limit": 3}).json()
    assert cities[0]["iata"] == "LIS" and cities[0]["city"] == "Lisbon"
    assert client.get("/v1/cities", params={"q": ""}).status_code == 422
    assert client.get("/v1/cities", params={"q": "a", "limit": 500}).status_code == 422
    meta = client.get("/v1/meta").json()
    assert "RUB" not in meta["currencies"] and "EUR" in meta["currencies"]
    assert meta["fx_as_of"]
