"""Appointment slots: 60 minutes, from each dentist's weekly local hours, stored as UTC instants.

A slot id is a uuid5 of the dentist and the start instant, so ids are stable without keeping the
slots in the database; only holds and bookings are stored. Local wall-clock times that do not
exist on a daylight-saving day are skipped, and an hour that happens twice is offered once.
"""

from datetime import UTC, date, datetime, timedelta
from uuid import UUID, uuid5
from zoneinfo import ZoneInfo

from dentabridge_api.seed import WEEKDAYS, ClinicSeed
from dentabridge_core import Slot

SLOT = timedelta(minutes=60)
LEAD_TIME = timedelta(hours=48)
HORIZON = timedelta(days=90)
_NAMESPACE = UUID("6f1d4a0e-5b7c-4d3e-9a52-0c8e1f2b7a64")


def slot_id(dentist_id: str, start: datetime) -> UUID:
    return uuid5(_NAMESPACE, f"{dentist_id}|{start.astimezone(UTC).isoformat()}")


def generate_slots(clinic: ClinicSeed, start: date, days: int) -> list[Slot]:
    zone = ZoneInfo(clinic.timezone)
    slots: list[Slot] = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        for dentist in clinic.dentists:
            for window in dentist.weekly_hours.get(WEEKDAYS[day.weekday()], ()):
                for hour in range(window.start.hour, window.end.hour):
                    local = datetime(day.year, day.month, day.day, hour, tzinfo=zone)
                    instant = local.astimezone(UTC)
                    if instant.astimezone(zone).replace(tzinfo=None) != local.replace(tzinfo=None):
                        continue  # this wall-clock hour does not exist today (clocks jumped)
                    slots.append(
                        Slot(
                            slot_id=slot_id(dentist.dentist_id, instant),
                            clinic_id=clinic.clinic_id,
                            dentist_id=dentist.dentist_id,
                            start=instant,
                            end=instant + SLOT,
                            status="available",
                        )
                    )
    return sorted(slots, key=lambda s: (s.start, s.dentist_id))


def bookable_window(now: datetime) -> tuple[datetime, datetime]:
    """Slots start no earlier than the lead time and no later than the horizon."""
    return now + LEAD_TIME, now + HORIZON
