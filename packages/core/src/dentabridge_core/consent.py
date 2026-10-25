"""Explicit consent for health data. Records carry no IP and no name."""

from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, model_validator

from dentabridge_core._base import Contract, Locale

ConsentPurpose = Literal["photo_analysis", "share_with_clinic"]


class ConsentRequest(Contract):
    purpose: ConsentPurpose
    health_data_explicit: Literal[True]
    text_version: str
    locale: Locale


class ConsentRecord(ConsentRequest):
    consent_id: UUID
    given_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def _expires_later(self) -> Self:
        if self.expires_at <= self.given_at:
            raise ValueError("expires_at must be after given_at")
        return self
