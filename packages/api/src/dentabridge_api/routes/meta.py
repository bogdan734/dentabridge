from typing import Literal

from fastapi import APIRouter, Depends, Request

from dentabridge_api.ratelimit import rate_limited
from dentabridge_api.schemas import Health, Meta
from dentabridge_core import ASSESSABLE, CONTRACT_VERSION, VERIFIED_PAIRS
from dentabridge_core.policy import texts

router = APIRouter(tags=["meta"], dependencies=[Depends(rate_limited("read"))])


@router.get("/health", operation_id="getHealth")
def get_health(request: Request) -> Health:
    state = request.app.state
    return Health(
        status="ok",
        contract_version=CONTRACT_VERSION,
        mode=state.settings.mode,
        vision_backend=state.vision.model.model_id,
    )


@router.get("/meta", operation_id="getMeta")
def get_meta(request: Request, locale: Literal["en", "ru"] = "en") -> Meta:
    state = request.app.state
    return Meta(
        uploads_enabled=state.settings.uploads == "on",
        assessable={m: sorted(codes, key=lambda c: c.value) for m, codes in ASSESSABLE.items()},
        texts=dict(texts(locale)),
        verified_pairs=sorted(VERIFIED_PAIRS),
        currencies=state.fx.currencies(),
        fx_as_of=state.fx.as_of,
    )
