from uuid import UUID

from fastapi import APIRouter, Depends, Request

from dentabridge_api.db.models import QuoteRow
from dentabridge_api.errors import ApiError
from dentabridge_api.ratelimit import rate_limited
from dentabridge_core import ErrorCode, Quote, QuoteRequest
from dentabridge_pricing.fares import UnknownAirport
from dentabridge_pricing.fx import CurrencyUnsupported
from dentabridge_pricing.quote import TripTooShort, build_quote

writing = APIRouter(tags=["quotes"], dependencies=[Depends(rate_limited("write"))])
reading = APIRouter(tags=["quotes"], dependencies=[Depends(rate_limited("read"))])


@writing.post("/quotes", status_code=201, operation_id="createQuote")
def create_quote(body: QuoteRequest, request: Request) -> Quote:
    state = request.app.state
    clinic = state.clinics.get(body.clinic_id)
    if clinic is None:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    summary = clinic.summary()
    offered = {t.treatment_code for t in summary.treatments}
    if not set(body.treatment_codes) <= offered:
        raise ApiError(422, ErrorCode.VALIDATION_ERROR, "treatment not offered by this clinic")
    if body.origin_iata == clinic.airport_iata:
        raise ApiError(422, ErrorCode.VALIDATION_ERROR, "origin is the clinic's own airport")
    try:
        quote = build_quote(
            body,
            summary,
            clinic.hotel_tier,
            state.fares,
            state.hotels,
            state.fx,
            state.clock.now(),
            fare_mdape=getattr(state.fares, "mdape", None),
        )
    except TripTooShort as error:
        raise ApiError(422, ErrorCode.TRIP_TOO_SHORT, str(error)) from error
    except CurrencyUnsupported as error:
        raise ApiError(422, ErrorCode.CURRENCY_UNSUPPORTED, str(error)) from error
    except UnknownAirport as error:
        raise ApiError(422, ErrorCode.UNKNOWN_AIRPORT, str(error)) from error
    with state.sessions.begin() as session:
        row = session.get(QuoteRow, str(quote.quote_id))
        if row is None:
            session.add(
                QuoteRow(
                    id=str(quote.quote_id),
                    clinic_id=clinic.clinic_id,
                    created_at=state.clock.now(),
                    valid_until=quote.valid_until,
                    quote_json=quote.model_dump_json(),
                )
            )
        else:  # the same request on the same day: the same quote, valid for a fresh day
            row.valid_until = quote.valid_until
            row.quote_json = quote.model_dump_json()
    return quote


@reading.get("/quotes/{quote_id}", operation_id="getQuote")
def get_quote(quote_id: UUID, request: Request) -> Quote:
    state = request.app.state
    with state.sessions() as session:
        row = session.get(QuoteRow, str(quote_id))
    if row is None:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    if row.valid_until <= state.clock.now():
        raise ApiError(410, ErrorCode.GONE)
    return Quote.model_validate_json(row.quote_json)
