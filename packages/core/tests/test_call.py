from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from dentabridge_core import (
    VERIFIED_PAIRS,
    AiDisclosure,
    CallSession,
    CallTokenRequest,
    CaptionEvent,
    ChatEvent,
    ReplayManifest,
)

NOW = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
DISCLOSURE: dict[str, object] = {
    "synthetic_voice": True,
    "machine_translation": True,
    "not_medical_interpreter": True,
    "must_acknowledge": True,
    "basis": "EU AI Act Art. 50",
    "text_key": "ai_disclosure.v1",
}


def _session(**changes: object) -> CallSession:
    data: dict[str, object] = {
        "session_id": UUID(int=1),
        "booking_id": UUID(int=2),
        "room": "db-room",
        "languages": {"patient": "ru", "dentist": "en"},
        "pair_status": "verified",
        "mode": "consecutive",
        "ai_disclosure": DISCLOSURE,
        "recording": False,
        "expires_at": NOW,
    }
    data.update(changes)
    return CallSession.model_validate(data)


def _chat(**changes: object) -> ChatEvent:
    data: dict[str, object] = {
        "message_id": UUID(int=3),
        "sender": "patient",
        "src_lang": "ru",
        "tgt_lang": "en",
        "src_text": "Болит зуб 36",
        "tgt_text": "Tooth 36 hurts",
        "translation_status": "ok",
        "digits_ok": True,
        "sent_at": NOW,
    }
    data.update(changes)
    return ChatEvent.model_validate(data)


@pytest.mark.parametrize(
    "field",
    ["synthetic_voice", "machine_translation", "not_medical_interpreter", "must_acknowledge"],
)
def test_disclosure_fields_literal_true(field: str) -> None:
    assert AiDisclosure.model_validate(DISCLOSURE).must_acknowledge is True
    with pytest.raises(ValidationError):
        AiDisclosure.model_validate({**DISCLOSURE, field: False})
    with pytest.raises(ValidationError):
        AiDisclosure.model_validate({k: v for k, v in DISCLOSURE.items() if k != field})


def test_session_never_records() -> None:
    assert _session().recording is False
    with pytest.raises(ValidationError):
        _session(recording=True)


def test_session_names_both_roles() -> None:
    with pytest.raises(ValidationError, match="both the patient and the dentist"):
        _session(languages={"patient": "ru"})
    with pytest.raises(ValidationError):
        _session(languages={"patient": "ru", "nurse": "en"})


def test_token_request_requires_acknowledgement() -> None:
    assert CallTokenRequest(role="dentist", ai_disclosure_acknowledged=True).role == "dentist"
    with pytest.raises(ValidationError):
        CallTokenRequest.model_validate({"role": "dentist", "ai_disclosure_acknowledged": False})
    with pytest.raises(ValidationError):
        CallTokenRequest.model_validate({"role": "dentist"})


@pytest.mark.parametrize(("length", "valid"), [(0, False), (1, True), (1000, True), (1001, False)])
def test_chat_event_length_bounds(length: int, valid: bool) -> None:
    if valid:
        assert len(_chat(src_text="x" * length).src_text) == length
    else:
        with pytest.raises(ValidationError):
            _chat(src_text="x" * length)


def test_chat_translation_matches_status() -> None:
    with pytest.raises(ValidationError, match="exactly when"):
        _chat(tgt_text=None)
    with pytest.raises(ValidationError, match="exactly when"):
        _chat(translation_status="failed")
    assert _chat(translation_status="failed", tgt_text=None).tgt_text is None


def _caption(**changes: object) -> CaptionEvent:
    data: dict[str, object] = {
        "utterance_id": "patient-1",
        "speaker": "patient",
        "src_lang": "ru",
        "tgt_lang": "en",
        "src_text": "Болит зуб 36",
        "tgt_text": "Tooth 36 hurts",
        "final": True,
        "digits_ok": True,
        "audio": "spoken",
        "t_speech_end_ms": 1200,
    }
    data.update(changes)
    return CaptionEvent.model_validate(data)


def test_a_caption_is_a_translation_unless_it_says_it_failed() -> None:
    assert _caption().translation_status == "ok"
    failed = {"translation_status": "failed", "tgt_text": "", "audio": "captions_only"}
    assert _caption(**failed).tgt_text == ""
    for wrong in ({"tgt_text": "Tooth 36 hurts"}, {"final": False}, {"audio": "spoken"}):
        with pytest.raises(ValidationError, match="failed translation"):
            _caption(**{**failed, **wrong})


def test_replay_p95_not_below_p50() -> None:
    base: dict[str, object] = {"replay_id": "demo", "tracks": [], "events": [], "n": 0}
    assert ReplayManifest.model_validate({**base, "latency_p50_ms": 1, "latency_p95_ms": 2}).n == 0
    with pytest.raises(ValidationError, match="p95"):
        ReplayManifest.model_validate({**base, "latency_p50_ms": 5, "latency_p95_ms": 2})


def test_verified_pairs_are_en_ru_only() -> None:
    assert frozenset({("en", "ru"), ("ru", "en")}) == VERIFIED_PAIRS
