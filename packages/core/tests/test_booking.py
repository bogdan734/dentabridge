from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from dentabridge_core import Booking, BookingRequest, ConsentRecord, MockTicket, Slot

START = datetime(2026, 11, 9, 10, 0, tzinfo=UTC)


def _slot(**changes: object) -> Slot:
    data: dict[str, object] = {
        "slot_id": UUID(int=1),
        "clinic_id": "demo-clinic",
        "dentist_id": "dr-demo",
        "start": START,
        "end": START + timedelta(hours=1),
        "status": "available",
    }
    data.update(changes)
    return Slot.model_validate(data)


@pytest.mark.parametrize(
    ("analysis", "consent", "valid"),
    [
        (None, None, True),
        (UUID(int=1), UUID(int=2), True),
        (UUID(int=1), None, False),
        (None, UUID(int=2), False),
    ],
)
def test_booking_request_share_pair(
    analysis: UUID | None, consent: UUID | None, valid: bool
) -> None:
    data = {
        "hold_id": UUID(int=7),
        "quote_id": UUID(int=8),
        "analysis_id": analysis,
        "share_consent_id": consent,
    }
    if valid:
        assert BookingRequest.model_validate(data).analysis_id == analysis
    else:
        with pytest.raises(ValidationError, match="together or not at all"):
            BookingRequest.model_validate(data)


def test_slot_ends_after_start() -> None:
    assert _slot().end > _slot().start
    with pytest.raises(ValidationError, match="ends after it starts"):
        _slot(end=START)


def test_booking_is_simulated() -> None:
    data = {
        "booking_id": UUID(int=9),
        "slot": _slot().model_dump(),
        "quote_id": UUID(int=8),
        "status": "confirmed",
        "simulated": True,
    }
    assert Booking.model_validate(data).shared_analysis_id is None
    with pytest.raises(ValidationError):
        Booking.model_validate({**data, "simulated": False})


@pytest.mark.parametrize("reference", ["DEMO-AB12", "TK-123", "demo-1", "DEMO-"])
def test_mock_ticket_reference_is_marked_demo(reference: str) -> None:
    data = {"kind": "flight", "reference": reference, "chip": "SIMULATED"}
    if reference == "DEMO-AB12":
        assert MockTicket.model_validate(data).chip == "SIMULATED"
    elif reference != "DEMO-":
        with pytest.raises(ValidationError):
            MockTicket.model_validate(data)


def test_consent_needs_explicit_health_data_and_expires_later() -> None:
    data: dict[str, object] = {
        "purpose": "share_with_clinic",
        "health_data_explicit": True,
        "text_version": "v1",
        "locale": "ru",
        "consent_id": UUID(int=5),
        "given_at": START,
        "expires_at": START + timedelta(hours=24),
    }
    assert ConsentRecord.model_validate(data).purpose == "share_with_clinic"
    with pytest.raises(ValidationError):
        ConsentRecord.model_validate({**data, "health_data_explicit": False})
    with pytest.raises(ValidationError):
        ConsentRecord.model_validate({**data, "purpose": "marketing"})
    with pytest.raises(ValidationError, match="expires_at"):
        ConsentRecord.model_validate({**data, "expires_at": START})
