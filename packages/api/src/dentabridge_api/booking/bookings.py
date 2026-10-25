"""Confirming and cancelling a simulated booking. Nothing here is a real reservation."""

import json
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dentabridge_api.booking.slots import SLOT
from dentabridge_api.db.models import AnalysisRow, BookingRow, ConsentRow, HoldRow, QuoteRow
from dentabridge_api.errors import ApiError
from dentabridge_api.seed import ClinicSeed
from dentabridge_core import (
    Booking,
    BookingRequest,
    ErrorCode,
    MockTicket,
    Quote,
    Slot,
)

SNAPSHOT_KEPT_AFTER_APPOINTMENT = timedelta(hours=24)


def _tickets(booking_id: UUID) -> list[dict[str, str]]:
    code = booking_id.hex[:8].upper()
    return [
        MockTicket(kind="flight", reference=f"DEMO-FLT-{code}", chip="SIMULATED").model_dump(),
        MockTicket(kind="hotel", reference=f"DEMO-HTL-{code}", chip="SIMULATED").model_dump(),
    ]


def to_contract(row: BookingRow) -> Booking:
    confirmed = row.status == "confirmed"
    return Booking(
        booking_id=UUID(row.id),
        slot=Slot(
            slot_id=UUID(row.slot_id),
            clinic_id=row.clinic_id,
            dentist_id=row.dentist_id,
            start=row.start,
            end=row.end,
            status="booked" if confirmed else "available",
        ),
        quote_id=UUID(row.quote_id),
        status="confirmed" if confirmed else "cancelled",
        tickets=tuple(MockTicket.model_validate(t) for t in json.loads(row.tickets_json)),
        shared_analysis_id=UUID(row.shared_analysis_id) if row.shared_analysis_id else None,
        simulated=True,
    )


def confirm(
    session: Session, req: BookingRequest, clinics: dict[str, ClinicSeed], now: datetime
) -> Booking:
    hold = session.get(HoldRow, str(req.hold_id))
    if hold is None:
        raise ApiError(404, ErrorCode.NOT_FOUND, "hold")
    if hold.state != "active" or hold.expires_at <= now:
        if hold.state == "active":
            hold.state = "expired"
        raise ApiError(409, ErrorCode.HOLD_EXPIRED)
    quote_row = session.get(QuoteRow, str(req.quote_id))
    if quote_row is None:
        raise ApiError(404, ErrorCode.NOT_FOUND, "quote")
    if quote_row.valid_until <= now:
        raise ApiError(410, ErrorCode.QUOTE_EXPIRED)
    quote = Quote.model_validate_json(quote_row.quote_json)
    if quote.request.clinic_id != hold.clinic_id:
        raise ApiError(409, ErrorCode.QUOTE_MISMATCH)
    local_day = hold.start.astimezone(ZoneInfo(clinics[hold.clinic_id].timezone)).date()
    if not quote.request.depart_date < local_day < quote.request.return_date:
        raise ApiError(409, ErrorCode.OUTSIDE_TRIP)

    snapshot: str | None = None
    if req.analysis_id is not None:
        consent = session.get(ConsentRow, str(req.share_consent_id))
        if consent is None or consent.purpose != "share_with_clinic" or consent.expires_at <= now:
            raise ApiError(403, ErrorCode.SHARE_CONSENT_REQUIRED)
        analysis = session.get(AnalysisRow, str(req.analysis_id))
        if analysis is None or analysis.expires_at <= now:
            raise ApiError(410, ErrorCode.ANALYSIS_EXPIRED)
        snapshot = analysis.result_json  # observations only: no photo is ever stored

    booking_id = uuid4()
    end = hold.start + SLOT
    row = BookingRow(
        id=str(booking_id),
        slot_id=hold.slot_id,
        clinic_id=hold.clinic_id,
        dentist_id=hold.dentist_id,
        start=hold.start,
        end=end,
        quote_id=str(req.quote_id),
        hold_id=str(req.hold_id),
        status="confirmed",
        tickets_json=json.dumps(_tickets(booking_id)),
        shared_analysis_id=str(req.analysis_id) if req.analysis_id else None,
        snapshot_json=snapshot,
        snapshot_purge_after=end + SNAPSHOT_KEPT_AFTER_APPOINTMENT if snapshot else None,
        created_at=now,
    )
    hold.state = "consumed"
    session.add(row)
    try:
        session.flush()
    except IntegrityError as error:
        raise ApiError(409, ErrorCode.SLOT_UNAVAILABLE) from error
    return to_contract(row)


def cancel(session: Session, booking_id: UUID) -> Booking:
    row = session.get(BookingRow, str(booking_id))
    if row is None:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    if row.status == "confirmed":
        row.status = "cancelled"
        row.snapshot_json = None  # the shared observations go with the booking
        row.snapshot_purge_after = None
    return to_contract(row)


def purge_snapshots(session: Session, now: datetime) -> None:
    """Shared observations are deleted 24 hours after the appointment ends."""
    for row in session.query(BookingRow).filter(BookingRow.snapshot_purge_after <= now):
        row.snapshot_json = None
        row.snapshot_purge_after = None
