"""Tables. Nothing here stores image bytes or a client address."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Dialect, Index, Integer, String, Text, TypeDecorator, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UtcDateTime(TypeDecorator[datetime]):
    """Aware UTC in Python; SQLite returns naive values, so they are tagged on the way out."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> Any:
        if value is not None and value.tzinfo is None:
            raise ValueError("a naive datetime reached the database")
        return None if value is None else value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class ConsentRow(Base):
    __tablename__ = "consents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    purpose: Mapped[str] = mapped_column(String(32))
    text_version: Mapped[str] = mapped_column(String(32))
    locale: Mapped[str] = mapped_column(String(2))
    granted_at: Mapped[datetime] = mapped_column(UtcDateTime())
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime())


class AnalysisRow(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    consent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime())
    result_json: Mapped[str] = mapped_column(Text)


class IdempotencyRow(Base):
    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    route: Mapped[str] = mapped_column(String(128), primary_key=True)
    body_sha256: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(16))  # "in_progress" | "done"
    status: Mapped[int] = mapped_column(Integer, default=0)
    body: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())


class QuoteRow(Base):
    __tablename__ = "quotes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    clinic_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    valid_until: Mapped[datetime] = mapped_column(UtcDateTime())
    quote_json: Mapped[str] = mapped_column(Text)


class HoldRow(Base):
    __tablename__ = "holds"
    __table_args__ = (
        Index(
            "ux_active_hold_per_slot",
            "slot_id",
            unique=True,
            sqlite_where=text("state = 'active'"),
            postgresql_where=text("state = 'active'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    slot_id: Mapped[str] = mapped_column(String(36))
    clinic_id: Mapped[str] = mapped_column(String(64))
    dentist_id: Mapped[str] = mapped_column(String(64))
    start: Mapped[datetime] = mapped_column(UtcDateTime())
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime())
    state: Mapped[str] = mapped_column(String(12))  # active | consumed | released | expired


class BookingRow(Base):
    __tablename__ = "bookings"
    __table_args__ = (
        Index(
            "ux_confirmed_booking_per_slot",
            "slot_id",
            unique=True,
            sqlite_where=text("status = 'confirmed'"),
            postgresql_where=text("status = 'confirmed'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    slot_id: Mapped[str] = mapped_column(String(36))
    clinic_id: Mapped[str] = mapped_column(String(64))
    dentist_id: Mapped[str] = mapped_column(String(64))
    start: Mapped[datetime] = mapped_column(UtcDateTime())
    end: Mapped[datetime] = mapped_column(UtcDateTime())
    quote_id: Mapped[str] = mapped_column(String(36))
    hold_id: Mapped[str] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(12))  # confirmed | cancelled
    tickets_json: Mapped[str] = mapped_column(Text)
    shared_analysis_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot_purge_after: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())


class CallSessionRow(Base):
    __tablename__ = "call_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True
    )  # uuid5 of the booking: one call each
    booking_id: Mapped[str] = mapped_column(String(36), unique=True)
    patient_language: Mapped[str] = mapped_column(String(16))
    dentist_language: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime())
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime())
