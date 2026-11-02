"""A call session per booking, and what the dentist's panel may see of the patient."""

from datetime import datetime, timedelta
from uuid import UUID, uuid5

from sqlalchemy.orm import Session

from dentabridge_api.booking.bookings import to_contract as booking_contract
from dentabridge_api.db.models import BookingRow, CallSessionRow
from dentabridge_api.errors import ApiError
from dentabridge_api.seed import ClinicSeed
from dentabridge_core import (
    VERIFIED_PAIRS,
    AiDisclosure,
    AnalysisResult,
    CallContext,
    CallSession,
    ErrorCode,
    Locale,
)

SESSION_AFTER_APPOINTMENT = timedelta(hours=2)
_NAMESPACE = UUID("9a2c6e1b-3d84-4f57-b0a9-5e7d2c8f1a36")
DISCLOSURE = AiDisclosure(
    synthetic_voice=True,
    machine_translation=True,
    not_medical_interpreter=True,
    must_acknowledge=True,
    basis="EU AI Act Art. 50",
    text_key="ai_disclosure.body",
)


def session_id_of(booking_id: str) -> str:
    return str(uuid5(_NAMESPACE, f"call|{booking_id}"))


def to_contract(row: CallSessionRow) -> CallSession:
    """An unverified pair gets captions only: no synthetic voice for a pair nobody measured."""
    verified = (row.patient_language, row.dentist_language) in VERIFIED_PAIRS
    return CallSession(
        session_id=UUID(row.id),
        booking_id=UUID(row.booking_id),
        room=f"db-{row.booking_id}",
        languages={"patient": row.patient_language, "dentist": row.dentist_language},
        pair_status="verified" if verified else "unverified",
        mode="consecutive" if verified else "captions_only",
        ai_disclosure=DISCLOSURE,
        recording=False,
        expires_at=row.expires_at,
    )


def open_session(
    db: Session,
    booking: BookingRow,
    patient_language: str,
    dentist_language: str,
    clinics: dict[str, ClinicSeed],
    now: datetime,
) -> tuple[CallSession, bool]:
    """The call of a booking, and whether this call created it. One session per booking: asking
    again returns the first one, with the languages it was opened with."""
    if booking.status != "confirmed":
        raise ApiError(409, ErrorCode.BOOKING_CANCELLED)
    existing = db.get(CallSessionRow, session_id_of(booking.id))
    if existing is not None:
        return to_contract(existing), False
    dentist = next(
        d for d in clinics[booking.clinic_id].dentists if d.dentist_id == booking.dentist_id
    )
    if patient_language == dentist_language or dentist_language not in dentist.languages:
        raise ApiError(
            422,
            ErrorCode.LANGUAGE_PAIR_UNSUPPORTED,
            "the dentist speaks "
            + ", ".join(dentist.languages)
            + "; the call translates between two languages",
        )
    row = CallSessionRow(
        id=session_id_of(booking.id),
        booking_id=booking.id,
        patient_language=patient_language,
        dentist_language=dentist_language,
        created_at=now,
        expires_at=booking.end + SESSION_AFTER_APPOINTMENT,
    )
    db.add(row)
    return to_contract(row), True


def live_session(
    db: Session, session_id: UUID, now: datetime, cancelled: ErrorCode
) -> tuple[CallSessionRow, BookingRow]:
    """The session and its booking, or the error: 404 unknown, `cancelled` for a cancelled
    booking (the caller picks 409 or 410), 410 once the session has ended."""
    row = db.get(CallSessionRow, str(session_id))
    if row is None:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    booking = db.get(BookingRow, row.booking_id)
    if booking is None:  # a session never outlives its booking row
        raise ApiError(404, ErrorCode.NOT_FOUND)
    if booking.status != "confirmed":
        raise ApiError(409 if cancelled is ErrorCode.BOOKING_CANCELLED else 410, cancelled)
    if row.expires_at <= now:
        raise ApiError(410, ErrorCode.GONE)
    return row, booking


def build_context(
    booking: BookingRow, session: CallSessionRow, snapshot: AnalysisResult | None
) -> CallContext:
    """Slot, the patient's language and, only if shared, the observations. Never an image."""
    locale: Locale = "ru" if session.patient_language == "ru" else "en"
    return CallContext(
        session_id=UUID(session.id),
        booking_id=UUID(booking.id),
        slot=booking_contract(booking).slot,
        patient_locale=locale,
        analysis=snapshot,
    )
