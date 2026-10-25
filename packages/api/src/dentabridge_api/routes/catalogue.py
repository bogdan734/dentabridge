from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from dentabridge_api.booking.catalogue import with_status
from dentabridge_api.errors import ApiError
from dentabridge_api.ratelimit import rate_limited
from dentabridge_api.schemas import City
from dentabridge_api.seed import ClinicSeed
from dentabridge_core import ClinicSummary, ErrorCode, Slot, TreatmentCode
from dentabridge_pricing.airports import search_cities

router = APIRouter(tags=["catalogue"], dependencies=[Depends(rate_limited("read"))])
MAX_SLOT_DAYS = 31


@router.get("/treatments", operation_id="listTreatments")
def list_treatments() -> list[TreatmentCode]:
    return list(TreatmentCode)


@router.get("/cities", operation_id="searchCities")
def search(
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=64)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[City]:
    return [
        City(iata=a.iata, city=a.municipality, airport=a.name, country_code=a.iso_country)
        for a in search_cities(q, limit, request.app.state.airports)
    ]


@router.get("/clinics", operation_id="listClinics")
def list_clinics(request: Request, treatment: TreatmentCode | None = None) -> list[ClinicSummary]:
    summaries = [c.summary() for c in request.app.state.clinics.values()]
    if treatment is None:
        return summaries
    return [c for c in summaries if any(t.treatment_code == treatment for t in c.treatments)]


@router.get("/clinics/{clinic_id}", operation_id="getClinic")
def get_clinic(clinic_id: str, request: Request) -> ClinicSummary:
    clinic: ClinicSeed | None = request.app.state.clinics.get(clinic_id)
    if clinic is None:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    return clinic.summary()


@router.get("/clinics/{clinic_id}/slots", operation_id="listSlots")
def list_slots(
    clinic_id: str,
    request: Request,
    from_: Annotated[date, Query(alias="from")],
    to: date,
) -> list[Slot]:
    state = request.app.state
    if clinic_id not in state.clinics:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    if to < from_ or (to - from_).days >= MAX_SLOT_DAYS:
        raise ApiError(
            422, ErrorCode.VALIDATION_ERROR, f"from..to must span 1-{MAX_SLOT_DAYS} days"
        )
    now = state.clock.now()
    with state.sessions() as session:
        return with_status(session, state.catalogue.window(clinic_id, from_, to, now), now)
