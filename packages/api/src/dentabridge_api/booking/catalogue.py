"""Slots with their live status, and lookup of a slot by id."""

from datetime import UTC, date, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from dentabridge_api.booking.slots import bookable_window, generate_slots
from dentabridge_api.db.models import BookingRow, HoldRow
from dentabridge_api.seed import ClinicSeed
from dentabridge_core import Slot


class SlotCatalogue:
    def __init__(self, clinics: dict[str, ClinicSeed]) -> None:
        self._clinics = clinics
        self._index: dict[str, Slot] = {}
        self._indexed_on: date | None = None

    def window(self, clinic_id: str, first: date, last: date, now: datetime) -> list[Slot]:
        """Bookable slots (lead time and horizon applied) starting from `first` up to `last`."""
        lo, hi = bookable_window(now)
        days = (last - first).days + 1
        return [
            s
            for s in generate_slots(self._clinics[clinic_id], first, days)
            if lo <= s.start <= hi
            and first <= s.start.astimezone(UTC).date() <= last + timedelta(days=1)
        ]

    def find(self, slot_id: UUID, now: datetime) -> Slot | None:
        today = now.astimezone(UTC).date()
        if self._indexed_on != today:
            self._index = {
                str(s.slot_id): s
                for clinic_id in self._clinics
                for s in self.window(clinic_id, today, today + timedelta(days=91), now)
            }
            self._indexed_on = today
        return self._index.get(str(slot_id))


def with_status(session: Session, slots: list[Slot], now: datetime) -> list[Slot]:
    if not slots:
        return []
    ids = [str(s.slot_id) for s in slots]
    held = set(
        session.scalars(
            select(HoldRow.slot_id).where(
                HoldRow.slot_id.in_(ids), HoldRow.state == "active", HoldRow.expires_at > now
            )
        )
    )
    booked = set(
        session.scalars(
            select(BookingRow.slot_id).where(
                BookingRow.slot_id.in_(ids), BookingRow.status == "confirmed"
            )
        )
    )
    return [
        s.model_copy(
            update={
                "status": "booked"
                if str(s.slot_id) in booked
                else "held"
                if str(s.slot_id) in held
                else "available"
            }
        )
        for s in slots
    ]
