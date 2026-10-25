"""A hold keeps one slot for ten minutes. Expiry is checked lazily against the clock."""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dentabridge_api.db.models import BookingRow, HoldRow
from dentabridge_api.errors import ApiError
from dentabridge_core import ErrorCode, Hold, Slot

HOLD_FOR = timedelta(minutes=10)


def create_hold(session: Session, slot: Slot, now: datetime) -> Hold:
    slot_key = str(slot.slot_id)
    session.execute(  # a hold that ran out gives the slot back before anyone asks for it
        update(HoldRow)
        .where(HoldRow.slot_id == slot_key, HoldRow.state == "active", HoldRow.expires_at <= now)
        .values(state="expired")
    )
    taken = session.scalar(
        select(BookingRow.id).where(
            BookingRow.slot_id == slot_key, BookingRow.status == "confirmed"
        )
    )
    if taken is not None:
        raise ApiError(409, ErrorCode.SLOT_UNAVAILABLE)
    row = HoldRow(
        id=str(uuid4()),
        slot_id=slot_key,
        clinic_id=slot.clinic_id,
        dentist_id=slot.dentist_id,
        start=slot.start,
        created_at=now,
        expires_at=now + HOLD_FOR,
        state="active",
    )
    session.add(row)
    try:
        session.flush()
    except IntegrityError as error:  # the partial unique index: someone else holds it
        raise ApiError(409, ErrorCode.SLOT_UNAVAILABLE) from error
    return Hold(hold_id=UUID(row.id), slot_id=slot.slot_id, expires_at=row.expires_at)


def release_hold(session: Session, hold_id: UUID, now: datetime) -> None:
    row = session.get(HoldRow, str(hold_id))
    if row is None:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    if row.state == "active":
        row.state = "released" if row.expires_at > now else "expired"
