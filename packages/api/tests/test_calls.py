from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from dentabridge_api.clock import FrozenClock
from dentabridge_api.testing import (
    CLINIC,
    an_analysis,
    book,
    code_of,
    consent,
    field_names,
    first_slot,
    hold,
    key,
    quote,
)
from dentabridge_core import CallContext, CallSession, CallToken, ErrorCode, ReplayManifest

GALATA = "galata-demo-dental"


def booked(client: TestClient, share: bool = False, clinic: str = CLINIC) -> dict[str, Any]:
    trip = quote(client, clinic_id=clinic)
    held = hold(client, first_slot(client, clinic)["slot_id"]).json()
    extra = (
        {
            "analysis_id": an_analysis(client),
            "share_consent_id": consent(client, "share_with_clinic"),
        }
        if share
        else {}
    )
    response = book(client, held["hold_id"], trip["quote_id"], **extra)
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


def start_call(
    client: TestClient, booking_id: str, patient: str = "ru", dentist: str = "en"
) -> Any:
    return client.post(
        f"/v1/bookings/{booking_id}/call",
        json={"patient_language": patient, "dentist_language": dentist},
        headers=key(),
    )


def token_for(client: TestClient, session_id: str, role: str) -> str:
    response = client.post(
        f"/v1/calls/{session_id}/tokens", json={"role": role, "ai_disclosure_acknowledged": True}
    )
    assert response.status_code == 201, response.text
    return str(CallToken.model_validate(response.json()).token)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_a_call_session_is_created_once_per_booking(make_client: Any) -> None:
    client = make_client()
    booking = booked(client)
    first = start_call(client, booking["booking_id"])
    again = start_call(client, booking["booking_id"])
    assert (first.status_code, again.status_code) == (201, 200)
    session = CallSession.model_validate(first.json())
    assert again.json() == first.json()
    assert session.room == f"db-{booking['booking_id']}"
    assert session.languages == {"patient": "ru", "dentist": "en"}
    assert session.recording is False
    texts = client.get("/v1/meta").json()["texts"]
    assert session.ai_disclosure.text_key in texts


def test_a_session_ends_two_hours_after_the_appointment(make_client: Any) -> None:
    client = make_client()
    booking = booked(client)
    session = CallSession.model_validate(start_call(client, booking["booking_id"]).json())
    end = datetime.fromisoformat(booking["slot"]["end"])
    assert session.expires_at == end + timedelta(hours=2)


@pytest.mark.parametrize(
    ("patient", "dentist", "pair", "mode"),
    [("ru", "en", "verified", "consecutive"), ("en", "pt", "unverified", "captions_only")],
)
def test_pair_status_decides_whether_the_voice_speaks(
    make_client: Any, patient: str, dentist: str, pair: str, mode: str
) -> None:
    client = make_client()
    booking = booked(client)
    session = CallSession.model_validate(
        start_call(client, booking["booking_id"], patient, dentist).json()
    )
    assert (session.pair_status, session.mode) == (pair, mode)


@pytest.mark.parametrize(("patient", "dentist"), [("en", "en"), ("ru", "tr")])
def test_language_pair_unsupported_422(make_client: Any, patient: str, dentist: str) -> None:
    client = make_client()
    booking = booked(
        client
    )  # the Lisbon dentists speak en, pt and es: nothing to translate, or no tr
    response = start_call(client, booking["booking_id"], patient, dentist)
    assert (response.status_code, code_of(response)) == (422, ErrorCode.LANGUAGE_PAIR_UNSUPPORTED)


def test_call_on_cancelled_booking_409(make_client: Any) -> None:
    client = make_client()
    booking = booked(client)
    session = start_call(client, booking["booking_id"]).json()
    client.post(f"/v1/bookings/{booking['booking_id']}/cancel")
    response = start_call(client, booking["booking_id"])
    assert (response.status_code, code_of(response)) == (409, ErrorCode.BOOKING_CANCELLED)
    tokens = client.post(
        f"/v1/calls/{session['session_id']}/tokens",
        json={"role": "patient", "ai_disclosure_acknowledged": True},
    )
    assert (tokens.status_code, code_of(tokens)) == (409, ErrorCode.BOOKING_CANCELLED)


def test_unknown_booking_and_session_404(make_client: Any) -> None:
    client = make_client()
    nobody = "00000000-0000-4000-8000-000000000000"
    assert start_call(client, nobody).status_code == 404
    response = client.post(
        f"/v1/calls/{nobody}/tokens", json={"role": "patient", "ai_disclosure_acknowledged": True}
    )
    assert (response.status_code, code_of(response)) == (404, ErrorCode.NOT_FOUND)


def test_token_requires_ack_422(make_client: Any) -> None:
    client = make_client()
    session = start_call(client, booked(client)["booking_id"]).json()
    url = f"/v1/calls/{session['session_id']}/tokens"
    refused = client.post(url, json={"role": "patient", "ai_disclosure_acknowledged": False})
    assert (refused.status_code, code_of(refused)) == (422, ErrorCode.DISCLOSURE_NOT_ACKNOWLEDGED)
    missing = client.post(url, json={"role": "patient"})
    assert (missing.status_code, code_of(missing)) == (422, ErrorCode.VALIDATION_ERROR)


def test_token_grants_room_and_identity(make_client: Any) -> None:
    client = make_client()
    booking = booked(client)
    session = start_call(client, booking["booking_id"]).json()
    response = client.post(
        f"/v1/calls/{session['session_id']}/tokens",
        json={"role": "patient", "ai_disclosure_acknowledged": True},
    )
    token = CallToken.model_validate(response.json())
    settings = client.app.state.settings
    claims = jwt.decode(
        token.token,
        settings.call_secret,
        algorithms=["HS256"],
        issuer="dentabridge",
        options={"verify_exp": False, "verify_nbf": False},
    )
    assert token.identity == f"patient-{session['session_id']}" == claims["sub"]
    assert claims["video"]["room"] == f"db-{booking['booking_id']}"
    assert token.livekit_url is None


def test_a_configured_livekit_url_is_returned_with_the_token(make_client: Any) -> None:
    client = make_client(
        livekit_url="wss://rtc.example.test", livekit_key="devkey", livekit_secret="s" * 40
    )
    session = start_call(client, booked(client)["booking_id"]).json()
    response = client.post(
        f"/v1/calls/{session['session_id']}/tokens",
        json={"role": "dentist", "ai_disclosure_acknowledged": True},
    )
    assert CallToken.model_validate(response.json()).livekit_url == "wss://rtc.example.test"


def test_context_requires_dentist_token(make_client: Any) -> None:
    client = make_client()
    session = start_call(client, booked(client)["booking_id"]).json()
    sid = session["session_id"]
    url = f"/v1/calls/{sid}/context"
    for headers in ({}, bearer("garbage"), {"Authorization": "Basic abc"}):
        response = client.get(url, headers=headers)
        assert (response.status_code, code_of(response)) == (401, ErrorCode.INVALID_CALL_TOKEN)
    patient = client.get(url, headers=bearer(token_for(client, sid, "patient")))
    assert (patient.status_code, code_of(patient)) == (403, ErrorCode.WRONG_ROLE)
    other = start_call(client, booked(client, clinic=GALATA)["booking_id"], "ru", "en").json()
    foreign = client.get(url, headers=bearer(token_for(client, other["session_id"], "dentist")))
    assert (foreign.status_code, code_of(foreign)) == (401, ErrorCode.INVALID_CALL_TOKEN)
    assert client.get(url, headers=bearer(token_for(client, sid, "dentist"))).status_code == 200


def test_context_without_share_has_no_analysis(make_client: Any) -> None:
    client = make_client()
    booking = booked(client)
    session = start_call(client, booking["booking_id"], "ru", "en").json()
    response = client.get(
        f"/v1/calls/{session['session_id']}/context",
        headers=bearer(token_for(client, session["session_id"], "dentist")),
    )
    context = CallContext.model_validate(response.json())
    assert context.analysis is None
    assert (str(context.booking_id), context.patient_locale) == (booking["booking_id"], "ru")
    assert context.slot.slot_id is not None


def test_context_with_a_shared_analysis_never_contains_image_bytes(make_client: Any) -> None:
    client = make_client()
    booking = booked(client, share=True)
    session = start_call(client, booking["booking_id"], "en", "pt").json()
    response = client.get(
        f"/v1/calls/{session['session_id']}/context",
        headers=bearer(token_for(client, session["session_id"], "dentist")),
    )
    context = CallContext.model_validate(response.json())
    assert context.analysis is not None
    assert str(context.analysis.analysis_id) == booking["shared_analysis_id"]
    photo_keys = [k for k in field_names(response.json()) if "image" in k and k != "image_quality"]
    assert not photo_keys, "observations only: no photo and no reference to one"


def test_context_is_gone_after_a_cancel(make_client: Any) -> None:
    client = make_client()
    cancelled = booked(client, share=True)
    session = start_call(client, cancelled["booking_id"]).json()
    token = token_for(client, session["session_id"], "dentist")
    client.post(f"/v1/bookings/{cancelled['booking_id']}/cancel")
    response = client.get(f"/v1/calls/{session['session_id']}/context", headers=bearer(token))
    assert (response.status_code, code_of(response)) == (410, ErrorCode.GONE)


def test_a_session_that_has_ended_is_gone_even_for_a_token_still_in_date(
    make_client: Any, clock: FrozenClock
) -> None:
    client = make_client()
    session = CallSession.model_validate(start_call(client, booked(client)["booking_id"]).json())
    sid = str(session.session_id)
    clock.set(session.expires_at - timedelta(hours=1))
    token = token_for(client, sid, "dentist")
    clock.set(session.expires_at + timedelta(seconds=1))
    gone = client.get(f"/v1/calls/{sid}/context", headers=bearer(token))
    assert (gone.status_code, code_of(gone)) == (410, ErrorCode.GONE)
    minting = client.post(
        f"/v1/calls/{sid}/tokens", json={"role": "dentist", "ai_disclosure_acknowledged": True}
    )
    assert (minting.status_code, code_of(minting)) == (410, ErrorCode.GONE)


def test_an_expired_token_is_refused(make_client: Any, clock: FrozenClock) -> None:
    client = make_client()
    session = start_call(client, booked(client)["booking_id"]).json()
    token = token_for(client, session["session_id"], "dentist")
    clock.set(clock.now() + timedelta(hours=2, seconds=1))
    response = client.get(f"/v1/calls/{session['session_id']}/context", headers=bearer(token))
    assert (response.status_code, code_of(response)) == (401, ErrorCode.INVALID_CALL_TOKEN)


@pytest.fixture
def replays(tmp_path: Path) -> Path:
    from dentabridge_core import CaptionEvent, ReplayTrack

    folder = tmp_path / "replays" / "demo"
    (folder / "audio").mkdir(parents=True)
    (folder / "audio" / "patient.ogg").write_bytes(b"OggS-patient")
    manifest = ReplayManifest(
        replay_id="demo",
        tracks=(ReplayTrack(role="patient", lang="ru", url="audio/patient.ogg"),),
        events=(
            CaptionEvent(
                utterance_id="u1",
                speaker="patient",
                src_lang="ru",
                tgt_lang="en",
                src_text="Болит зуб",
                tgt_text="My tooth hurts",
                final=True,
                digits_ok=True,
                audio="spoken",
                t_speech_end_ms=1000,
                t_first_audio_ms=1900,
            ),
        ),
        latency_p50_ms=900,
        latency_p95_ms=1400,
        n=1,
    )
    (folder / "manifest.json").write_text(manifest.model_dump_json())
    return tmp_path / "replays"


def test_replay_manifest_validates_against_core(make_client: Any, replays: Path) -> None:
    client = make_client(replays_dir=replays)
    response = client.get("/v1/replays/demo")
    assert response.status_code == 200
    assert ReplayManifest.model_validate(response.json()).replay_id == "demo"


def test_replay_audio_is_served_as_ogg(make_client: Any, replays: Path) -> None:
    client = make_client(replays_dir=replays)
    response = client.get("/v1/replays/demo/audio/patient.ogg")
    assert (response.status_code, response.headers["content-type"], response.content) == (
        200,
        "audio/ogg",
        b"OggS-patient",
    )


@pytest.mark.parametrize(
    "path",
    [
        "/v1/replays/nope",
        "/v1/replays/demo/audio/missing.ogg",
        "/v1/replays/demo/audio/..%2Fmanifest.json",
        "/v1/replays/..%2Fdemo",
    ],
)
def test_unknown_replays_and_path_tricks_404(make_client: Any, replays: Path, path: str) -> None:
    response = make_client(replays_dir=replays).get(path)
    assert (response.status_code, code_of(response)) == (404, ErrorCode.NOT_FOUND)


def test_a_corrupt_manifest_is_a_server_error_not_a_leak(make_client: Any, replays: Path) -> None:
    (replays / "demo" / "manifest.json").write_text("{not json")
    response = make_client(replays_dir=replays).get("/v1/replays/demo")
    assert (response.status_code, code_of(response)) == (500, ErrorCode.INTERNAL_ERROR)
