from fastapi import APIRouter, Depends, Request, Response

from dentabridge_api.errors import ApiError
from dentabridge_api.ratelimit import rate_limited
from dentabridge_api.schemas import SampleImage
from dentabridge_core import ErrorCode

router = APIRouter(tags=["samples"], dependencies=[Depends(rate_limited("read"))])


@router.get("/samples", operation_id="listSamples")
def list_samples(request: Request) -> list[SampleImage]:
    return [
        SampleImage(id=s.id, attribution=s.attribution, licence=s.licence)
        for s in request.app.state.samples.all()
    ]


@router.get(
    "/samples/{sample_id}/image",
    operation_id="getSampleImage",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def get_sample_image(sample_id: str, request: Request) -> Response:
    sample = request.app.state.samples.get(sample_id)
    if sample is None:
        raise ApiError(404, ErrorCode.NOT_FOUND)
    return Response(sample.jpeg(), media_type="image/jpeg")
