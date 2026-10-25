import pytest
from pydantic import ValidationError

from dentabridge_core import ErrorCode, ErrorResponse

# Every code in the PLAN.md "API surface" table and its "Common errors" line, plus not_found, gone.
API_TABLE_CODES = {
    "image_unreadable",
    "consent_required",
    "uploads_disabled",
    "consent_expired",
    "payload_too_large",
    "unsupported_media_type",
    "backend_unavailable",
    "not_found",
    "gone",
    "trip_too_short",
    "currency_unsupported",
    "unknown_airport",
    "slot_unavailable",
    "hold_expired",
    "quote_mismatch",
    "outside_trip",
    "quote_expired",
    "analysis_expired",
    "share_consent_required",
    "booking_cancelled",
    "language_pair_unsupported",
    "disclosure_not_acknowledged",
    "invalid_call_token",
    "wrong_role",
    "idempotency_key_required",
    "idempotency_in_progress",
    "idempotency_key_reuse",
    "validation_error",
    "rate_limited",
    "internal_error",
}


def test_error_codes_cover_api_table() -> None:
    assert {code.value for code in ErrorCode} == API_TABLE_CODES


def test_error_response_shape() -> None:
    response = ErrorResponse(
        code=ErrorCode.RATE_LIMITED,
        message_key="errors.rate_limited",
        request_id="r-1",
        retry_after_s=3,
    )
    assert response.detail is None
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate(
            {"code": "rate_limited", "message_key": "k", "request_id": "r", "retry_after_s": -1}
        )
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate({"code": "teapot", "message_key": "k", "request_id": "r"})
