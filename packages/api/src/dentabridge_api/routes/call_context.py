from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request

from dentabridge_api.booking.bookings import purge_snapshots
from dentabridge_api.call_tokens import InvalidCallToken, read_call_token
from dentabridge_api.calls import build_context, live_session
from dentabridge_api.errors import ApiError
from dentabridge_api.ratelimit import rate_limited
from dentabridge_core import AnalysisResult, CallContext, ErrorCode

router = APIRouter(tags=["calls"], dependencies=[Depends(rate_limited("read"))])


@router.get("/calls/{session_id}/context", operation_id="getCallContext")
def get_context(
    session_id: UUID, request: Request, authorization: Annotated[str | None, Header()] = None
) -> CallContext:
    state = request.app.state
    now = state.clock.now()
    scheme, _, token = (authorization or "").partition(" ")
    try:
        if scheme != "Bearer":
            raise InvalidCallToken("a Bearer call token is required")
        claims = read_call_token(token, state.settings, now)
    except InvalidCallToken as error:
        raise ApiError(401, ErrorCode.INVALID_CALL_TOKEN) from error
    if claims.session_id != session_id:
        raise ApiError(401, ErrorCode.INVALID_CALL_TOKEN)
    if claims.role != "dentist":
        raise ApiError(403, ErrorCode.WRONG_ROLE)
    with state.sessions.begin() as db:
        purge_snapshots(db, now)
        row, booking = live_session(db, session_id, now, ErrorCode.GONE)
        snapshot = (
            AnalysisResult.model_validate_json(booking.snapshot_json)
            if booking.snapshot_json
            else None
        )
        return build_context(booking, row, snapshot)
