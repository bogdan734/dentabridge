"""Slots, holds and simulated bookings. Nothing here is a real reservation."""

from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from dentabridge_core._base import Contract, Slug

SlotStatus = Literal["available", "held", "booked"]


class Slot(Contract):
    slot_id: UUID
    clinic_id: Slug
    dentist_id: Slug
    start: AwareDatetime
    end: AwareDatetime
    status: SlotStatus

    @model_validator(mode="after")
    def _ends_after_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError("a slot ends after it starts")
        return self


class Hold(Contract):
    hold_id: UUID
    slot_id: UUID
    expires_at: AwareDatetime


class BookingRequest(Contract):
    hold_id: UUID
    quote_id: UUID
    analysis_id: UUID | None = None
    share_consent_id: UUID | None = None

    @model_validator(mode="after")
    def _share_pair(self) -> Self:
        if (self.analysis_id is None) != (self.share_consent_id is None):
            raise ValueError("analysis_id and share_consent_id are given together or not at all")
        return self


class MockTicket(Contract):
    kind: Literal["flight", "hotel"]
    reference: str = Field(pattern=r"^DEMO-[A-Z0-9-]+$")
    chip: Literal["SIMULATED"]


class Booking(Contract):
    booking_id: UUID
    slot: Slot
    quote_id: UUID
    status: Literal["confirmed", "cancelled"]
    tickets: tuple[MockTicket, ...] = ()
    shared_analysis_id: UUID | None = None
    simulated: Literal[True]
