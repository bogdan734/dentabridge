from datetime import timedelta
from typing import Annotated
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Form, Request, Response, UploadFile

from dentabridge_api.analysis_service import run_analysis
from dentabridge_api.db.models import AnalysisRow, ConsentRow
from dentabridge_api.demo.fixture_vision import UnknownImage
from dentabridge_api.errors import ApiError
from dentabridge_api.idempotency import IdempotentRoute
from dentabridge_api.purge import purge_expired
from dentabridge_api.ratelimit import rate_limited
from dentabridge_api.schemas import AnalyseSampleRequest
from dentabridge_api.uploads import sanitise_upload
from dentabridge_core import AnalysisResult, ErrorCode, Locale, Modality, SanitisedImage
from dentabridge_ml.serving.parse import ParseError

reading = APIRouter(tags=["analyses"], dependencies=[Depends(rate_limited("read"))])
creating = APIRouter(
    tags=["analyses"], route_class=IdempotentRoute, dependencies=[Depends(rate_limited("analysis"))]
)


async def _analyse_and_store(
    request: Request, image: SanitisedImage, modality: Modality, locale: Locale, consent: str | None
) -> AnalysisResult:
    state = request.app.state
    try:
        result = await run_analysis(
            image,
            modality,
            locale,
            state.vision,
            state.clock,
            timedelta(hours=state.settings.result_ttl_h),
            state.urgency_table,
        )
    except (httpx.HTTPError, ParseError, UnknownImage, OSError) as error:
        raise ApiError(503, ErrorCode.BACKEND_UNAVAILABLE, type(error).__name__) from error
    with state.sessions.begin() as session:
        purge_expired(session, state.clock.now())
        session.add(
            AnalysisRow(
                id=str(result.analysis_id),
                consent_id=consent,
                created_at=result.created_at,
                expires_at=result.expires_at,
                result_json=result.model_dump_json(),
            )
        )
    return result


@creating.post("/analyses", status_code=201, operation_id="createAnalysis")
async def create_analysis(
    request: Request,
    file: UploadFile,
    consent_id: Annotated[UUID, Form()],
    modality: Annotated[Modality, Form()],
    locale: Annotated[Locale, Form()],
) -> AnalysisResult:
    state = request.app.state
    if state.settings.uploads == "off":
        raise ApiError(403, ErrorCode.UPLOADS_DISABLED)
    with state.sessions() as session:
        consent = session.get(ConsentRow, str(consent_id))
    if consent is None or consent.purpose != "photo_analysis":
        raise ApiError(403, ErrorCode.CONSENT_REQUIRED)
    if consent.expires_at <= state.clock.now():
        raise ApiError(410, ErrorCode.CONSENT_EXPIRED)
    image = sanitise_upload(await file.read(), file.content_type or "")
    return await _analyse_and_store(request, image, modality, locale, str(consent_id))


@creating.post("/analyses/sample", status_code=201, operation_id="createSampleAnalysis")
async def create_sample_analysis(body: AnalyseSampleRequest, request: Request) -> AnalysisResult:
    sample = request.app.state.samples.get(body.sample_id)
    if sample is None:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    return await _analyse_and_store(request, sample.sanitised(), sample.modality, body.locale, None)


@reading.get("/analyses/{analysis_id}", operation_id="getAnalysis")
def get_analysis(analysis_id: UUID, request: Request) -> AnalysisResult:
    state = request.app.state
    with state.sessions.begin() as session:
        row = session.get(AnalysisRow, str(analysis_id))
        if row is None:
            raise ApiError(404, ErrorCode.NOT_FOUND)
        if row.expires_at <= state.clock.now():
            session.delete(row)
            raise ApiError(410, ErrorCode.GONE)
        return AnalysisResult.model_validate_json(row.result_json)


@reading.delete("/analyses/{analysis_id}", status_code=204, operation_id="deleteAnalysis")
def delete_analysis(analysis_id: UUID, request: Request) -> Response:
    with request.app.state.sessions.begin() as session:
        row = session.get(AnalysisRow, str(analysis_id))
        if row is None:
            raise ApiError(404, ErrorCode.NOT_FOUND)
        session.delete(row)
    return Response(status_code=204)
