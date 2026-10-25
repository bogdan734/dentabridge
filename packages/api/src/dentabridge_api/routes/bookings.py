from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response

from dentabridge_api.booking import bookings as service
from dentabridge_api.booking.holds import create_hold, release_hold
from dentabridge_api.db.models import BookingRow
from dentabridge_api.errors import ApiError
from dentabridge_api.idempotency import IdempotentRoute
from dentabridge_api.ratelimit import rate_limited
from dentabridge_api.schemas import HoldRequest
from dentabridge_core import Booking, BookingRequest, ErrorCode, Hold

writing = APIRouter(tags=["booking"], dependencies=[Depends(rate_limited("write"))])
idempotent = APIRouter(
    tags=["booking"], route_class=IdempotentRoute, dependencies=[Depends(rate_limited("write"))]
)
reading = APIRouter(tags=["booking"], dependencies=[Depends(rate_limited("read"))])


@idempotent.post("/holds", status_code=201, operation_id="createHold")
def post_hold(body: HoldRequest, request: Request) -> Hold:
    state = request.app.state
    now = state.clock.now()
    slot = state.catalogue.find(body.slot_id, now)
    if slot is None:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    with state.sessions.begin() as session:
        return create_hold(session, slot, now)


@writing.delete("/holds/{hold_id}", status_code=204, operation_id="releaseHold")
def delete_hold(hold_id: UUID, request: Request) -> Response:
    state = request.app.state
    with state.sessions.begin() as session:
        release_hold(session, hold_id, state.clock.now())
    return Response(status_code=204)


@idempotent.post("/bookings", status_code=201, operation_id="createBooking")
def post_booking(body: BookingRequest, request: Request) -> Booking:
    state = request.app.state
    now = state.clock.now()
    with state.sessions.begin() as session:
        service.purge_snapshots(session, now)
        return service.confirm(session, body, state.clinics, now)


@reading.get("/bookings/{booking_id}", operation_id="getBooking")
def get_booking(booking_id: UUID, request: Request) -> Booking:
    with request.app.state.sessions() as session:
        row = session.get(BookingRow, str(booking_id))
        if row is None:
            raise ApiError(404, ErrorCode.NOT_FOUND)
        return service.to_contract(row)


@writing.post("/bookings/{booking_id}/cancel", operation_id="cancelBooking")
def cancel_booking(booking_id: UUID, request: Request) -> Booking:
    with request.app.state.sessions.begin() as session:
        return service.cancel(session, booking_id)
