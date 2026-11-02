from datetime import timedelta

from fastapi import APIRouter, Depends, Request

from dentabridge_api.db.models import ConsentRow
from dentabridge_api.errors import ApiError
from dentabridge_api.ids import new_id
from dentabridge_api.purge import purge_expired
from dentabridge_api.ratelimit import rate_limited
from dentabridge_core import ConsentRecord, ConsentRequest, ErrorCode
from dentabridge_core.policy import texts

router = APIRouter(tags=["consents"], dependencies=[Depends(rate_limited("write"))])


@router.post("/consents", status_code=201, operation_id="createConsent")
def create_consent(body: ConsentRequest, request: Request) -> ConsentRecord:
    state = request.app.state
    if f"consent.{body.purpose}.{body.text_version}" not in texts(body.locale):
        raise ApiError(
            422, ErrorCode.VALIDATION_ERROR, "text_version is not a current consent text"
        )
    now = state.clock.now()
    record = ConsentRecord(
        **body.model_dump(),
        consent_id=new_id(),
        given_at=now,
        expires_at=now + timedelta(hours=state.settings.result_ttl_h),
    )
    with state.sessions.begin() as session:
        purge_expired(session, now)
        session.add(
            ConsentRow(
                id=str(record.consent_id),
                purpose=record.purpose,
                text_version=record.text_version,
                locale=record.locale,
                granted_at=record.given_at,
                expires_at=record.expires_at,
            )
        )
    return record
