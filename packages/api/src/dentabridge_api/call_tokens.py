"""Call tokens: a LiveKit access token the room accepts, read back by the API as who is calling.

Signed with the LiveKit key and secret when they are configured, else with `call_secret`: demo mode
has no live room, but the dentist's panel still has to prove which side of which call it is. The
times come from the injected clock, so a fixture recorded twice is the same bytes.
"""

from datetime import datetime, timedelta
from uuid import UUID

import jwt
from livekit.api import AccessToken, VideoGrants
from livekit.protocol.room import RoomConfiguration
from pydantic import BaseModel, ConfigDict

from dentabridge_api.settings import Settings
from dentabridge_core import CallRoomMetadata, CallSession, CallToken, Role

TOKEN_TTL = timedelta(hours=2)
DEMO_ISSUER = "dentabridge"
_ROLES: tuple[Role, ...] = ("patient", "dentist")


class InvalidCallToken(Exception):
    pass


class CallClaims(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: UUID
    role: Role
    identity: str


def _signing(settings: Settings) -> tuple[str, str]:
    """(issuer, secret): LiveKit's own key pair when there is a room to join."""
    if settings.livekit_key and settings.livekit_secret:
        return settings.livekit_key, settings.livekit_secret
    return DEMO_ISSUER, settings.call_secret


def glossary_id(session: CallSession) -> str:
    return "dental-" + "-".join(sorted(session.languages.values()))


def mint_token(session: CallSession, role: Role, settings: Settings, now: datetime) -> CallToken:
    """Join one room, publish the microphone and data, subscribe; the room opens with the call
    metadata the voice worker routes by."""
    issuer, secret = _signing(settings)
    identity = f"{role}-{session.session_id}"
    metadata = CallRoomMetadata(
        session_id=session.session_id,
        languages=session.languages,
        mode=session.mode,
        glossary_id=glossary_id(session),
    )
    access = (
        AccessToken(issuer, secret)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=session.room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
                can_publish_sources=["microphone"],
            )
        )
        .with_room_config(RoomConfiguration(metadata=metadata.model_dump_json()))
    )
    expires_at = now + TOKEN_TTL
    claims = {
        **access.claims.asdict(),
        "sub": identity,
        "iss": issuer,
        "nbf": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return CallToken(
        token=jwt.encode(claims, secret, algorithm="HS256"),
        livekit_url=settings.livekit_url,
        identity=identity,
        expires_at=expires_at,
    )


def read_call_token(token: str, settings: Settings, now: datetime) -> CallClaims:
    issuer, secret = _signing(settings)
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],  # never "none"
            issuer=issuer,
            options={
                "require": ["exp", "nbf", "sub", "iss"],
                "verify_exp": False,
                "verify_nbf": False,
            },
        )
    except jwt.PyJWTError as error:
        raise InvalidCallToken(str(error)) from error
    if not claims["nbf"] <= now.timestamp() < claims["exp"]:  # the injected clock, not the wall
        raise InvalidCallToken("outside its validity window")
    role, _, session = str(claims["sub"]).partition("-")
    try:
        if role not in _ROLES:
            raise ValueError(role)
        return CallClaims(session_id=UUID(session), role=role, identity=str(claims["sub"]))
    except ValueError as error:
        raise InvalidCallToken("not a call identity") from error
