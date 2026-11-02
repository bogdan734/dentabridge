from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import jwt
import pytest

from dentabridge_api.call_tokens import CallClaims, InvalidCallToken, mint_token, read_call_token
from dentabridge_api.settings import Settings
from dentabridge_core import AiDisclosure, CallRoomMetadata, CallSession

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)
SESSION_ID = UUID("11111111-1111-4111-8111-111111111111")
BOOKING_ID = UUID("22222222-2222-4222-8222-222222222222")
LIVEKIT = {
    "livekit_url": "wss://rtc.example.test",
    "livekit_key": "devkey",
    "livekit_secret": "s" * 40,
}


def _session() -> CallSession:
    return CallSession(
        session_id=SESSION_ID,
        booking_id=BOOKING_ID,
        room=f"db-{BOOKING_ID}",
        languages={"patient": "ru", "dentist": "en"},
        pair_status="verified",
        mode="consecutive",
        ai_disclosure=AiDisclosure(
            synthetic_voice=True,
            machine_translation=True,
            not_medical_interpreter=True,
            must_acknowledge=True,
            basis="EU AI Act Art. 50",
            text_key="ai_disclosure.body",
        ),
        recording=False,
        expires_at=NOW + timedelta(days=9),
    )


def _claims(token: str, secret: str, issuer: str) -> dict[str, Any]:
    decoded: dict[str, Any] = jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        issuer=issuer,
        options={"verify_exp": False, "verify_nbf": False},
    )
    return decoded


def test_a_token_reads_back_as_the_role_and_session_it_was_minted_for() -> None:
    settings = Settings()
    token = mint_token(_session(), "dentist", settings, NOW)
    assert token.identity == f"dentist-{SESSION_ID}"
    assert token.expires_at == NOW + timedelta(hours=2)
    assert token.livekit_url is None, "demo mode has no live room"
    assert read_call_token(token.token, settings, NOW) == CallClaims(
        session_id=SESSION_ID, role="dentist", identity=token.identity
    )


def test_the_token_grants_one_room_the_microphone_and_data_only() -> None:
    settings = Settings()
    claims = _claims(
        mint_token(_session(), "patient", settings, NOW).token, settings.call_secret, "dentabridge"
    )
    assert (claims["sub"], claims["nbf"], claims["exp"]) == (
        f"patient-{SESSION_ID}",
        int(NOW.timestamp()),
        int((NOW + timedelta(hours=2)).timestamp()),
    )
    assert claims["video"] == {
        "roomJoin": True,
        "room": f"db-{BOOKING_ID}",
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
        "canPublishSources": ["microphone"],
    }


def test_the_room_is_created_with_the_call_metadata_the_voice_worker_reads() -> None:
    settings = Settings()
    claims = _claims(
        mint_token(_session(), "patient", settings, NOW).token, settings.call_secret, "dentabridge"
    )
    metadata = CallRoomMetadata.model_validate_json(claims["roomConfig"]["metadata"])
    assert metadata == CallRoomMetadata(
        session_id=SESSION_ID,
        languages={"patient": "ru", "dentist": "en"},
        mode="consecutive",
        glossary_id="dental-en-ru",
    )


def test_a_configured_livekit_signs_the_token_and_gives_its_url() -> None:
    settings = Settings.model_validate(LIVEKIT)
    token = mint_token(_session(), "dentist", settings, NOW)
    assert token.livekit_url == "wss://rtc.example.test"
    assert _claims(token.token, "s" * 40, "devkey")["sub"] == f"dentist-{SESSION_ID}"
    assert read_call_token(token.token, settings, NOW).role == "dentist"


@pytest.mark.parametrize(
    "partial", [{"livekit_url": "wss://x.test"}, {"livekit_key": "k", "livekit_secret": "s" * 40}]
)
def test_livekit_settings_are_all_or_nothing(partial: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="livekit"):
        Settings.model_validate(partial)


def test_a_token_lasts_two_hours_to_the_second() -> None:
    settings = Settings()
    token = mint_token(_session(), "dentist", settings, NOW).token
    read_call_token(token, settings, NOW + timedelta(hours=2) - timedelta(seconds=1))
    with pytest.raises(InvalidCallToken):
        read_call_token(token, settings, NOW + timedelta(hours=2))
    with pytest.raises(InvalidCallToken):
        read_call_token(token, settings, NOW - timedelta(seconds=1))


def test_forged_unsigned_and_foreign_tokens_are_refused() -> None:
    settings = Settings()
    good = mint_token(_session(), "dentist", settings, NOW)
    payload = _claims(good.token, settings.call_secret, "dentabridge")
    refused = [
        "not a token",
        jwt.encode(payload, "another secret that is long enough to sign", algorithm="HS256"),
        jwt.encode(payload, "", algorithm="none"),
        jwt.encode({**payload, "iss": "someone-else"}, settings.call_secret, algorithm="HS256"),
        jwt.encode({**payload, "sub": "dentist"}, settings.call_secret, algorithm="HS256"),
        jwt.encode(
            {**payload, "sub": "admin-" + str(SESSION_ID)}, settings.call_secret, algorithm="HS256"
        ),
    ]
    for token in refused:
        with pytest.raises(InvalidCallToken):
            read_call_token(token, settings, NOW)
