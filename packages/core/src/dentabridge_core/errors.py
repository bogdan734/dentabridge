"""Every error the API can return. Responses are always ErrorResponse plus X-Request-Id."""

from enum import StrEnum

from pydantic import Field

from dentabridge_core._base import Contract


class ErrorCode(StrEnum):
    IMAGE_UNREADABLE = "image_unreadable"
    CONSENT_REQUIRED = "consent_required"
    UPLOADS_DISABLED = "uploads_disabled"
    CONSENT_EXPIRED = "consent_expired"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    NOT_FOUND = "not_found"
    GONE = "gone"
    TRIP_TOO_SHORT = "trip_too_short"
    CURRENCY_UNSUPPORTED = "currency_unsupported"
    UNKNOWN_AIRPORT = "unknown_airport"
    SLOT_UNAVAILABLE = "slot_unavailable"
    HOLD_EXPIRED = "hold_expired"
    QUOTE_MISMATCH = "quote_mismatch"
    OUTSIDE_TRIP = "outside_trip"
    QUOTE_EXPIRED = "quote_expired"
    ANALYSIS_EXPIRED = "analysis_expired"
    SHARE_CONSENT_REQUIRED = "share_consent_required"
    BOOKING_CANCELLED = "booking_cancelled"
    LANGUAGE_PAIR_UNSUPPORTED = "language_pair_unsupported"
    DISCLOSURE_NOT_ACKNOWLEDGED = "disclosure_not_acknowledged"
    INVALID_CALL_TOKEN = "invalid_call_token"
    WRONG_ROLE = "wrong_role"
    IDEMPOTENCY_KEY_REQUIRED = "idempotency_key_required"
    IDEMPOTENCY_IN_PROGRESS = "idempotency_in_progress"
    IDEMPOTENCY_KEY_REUSE = "idempotency_key_reuse"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"


class ErrorResponse(Contract):
    code: ErrorCode
    message_key: str
    detail: str | None = None
    request_id: str
    retry_after_s: int | None = Field(default=None, ge=0)
