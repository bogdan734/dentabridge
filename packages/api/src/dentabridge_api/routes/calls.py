from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response

from dentabridge_api.call_tokens import mint_token
from dentabridge_api.calls import live_session, open_session, to_contract
from dentabridge_api.db.models import BookingRow
from dentabridge_api.errors import ApiError
from dentabridge_api.idempotency import IdempotentRoute
from dentabridge_api.ratelimit import rate_limited
from dentabridge_api.schemas import CallSessionRequest
from dentabridge_core import CallSession, CallToken, CallTokenRequest, ErrorCode

idempotent = APIRouter(
    tags=["calls"], route_class=IdempotentRoute, dependencies=[Depends(rate_limited("write"))]
)
writing = APIRouter(tags=["calls"], dependencies=[Depends(rate_limited("write"))])


@idempotent.post("/bookings/{booking_id}/call", status_code=201, operation_id="createCallSession")
def post_call(
    booking_id: UUID, body: CallSessionRequest, request: Request, response: Response
) -> CallSession:
    state = request.app.state
    with state.sessions.begin() as db:
        booking = db.get(BookingRow, str(booking_id))
        if booking is None:
            raise ApiError(404, ErrorCode.NOT_FOUND)
        session, created = open_session(
            db,
            booking,
            body.patient_language,
            body.dentist_language,
            state.clinics,
            state.clock.now(),
        )
    if not created:
        response.status_code = 200
    return session


@writing.post("/calls/{session_id}/tokens", status_code=201, operation_id="createCallToken")
def post_token(session_id: UUID, body: CallTokenRequest, request: Request) -> CallToken:
    state = request.app.state
    now = state.clock.now()
    with state.sessions() as db:
        row, _ = live_session(db, session_id, now, ErrorCode.BOOKING_CANCELLED)
        session = to_contract(row)
    return mint_token(session, body.role, state.settings, now)
