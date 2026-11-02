"""Call room objects: AI disclosure, session, tokens, live captions, chat and replays."""

from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from dentabridge_core._base import Contract, LanguageTag, Locale, Role
from dentabridge_core.analysis import AnalysisResult
from dentabridge_core.booking import Slot

PairStatus = Literal["verified", "unverified"]
CallMode = Literal["consecutive", "captions_only"]
AudioMode = Literal["spoken", "captions_only"]
TranslationStatus = Literal["ok", "failed"]

# Pairs whose quality the voice harness has measured. Every other pair is flagged "unverified".
VERIFIED_PAIRS: frozenset[tuple[str, str]] = frozenset({("en", "ru"), ("ru", "en")})


def _both_roles(languages: dict[Role, str]) -> None:
    if set(languages) != {"patient", "dentist"}:
        raise ValueError("languages must name both the patient and the dentist")


class AiDisclosure(Contract):
    """What the user must acknowledge before joining (EU AI Act Art. 50)."""

    synthetic_voice: Literal[True]
    machine_translation: Literal[True]
    not_medical_interpreter: Literal[True]
    must_acknowledge: Literal[True]
    basis: Literal["EU AI Act Art. 50"]
    text_key: str


class CallSession(Contract):
    session_id: UUID
    booking_id: UUID
    room: str
    languages: dict[Role, LanguageTag]
    pair_status: PairStatus
    mode: CallMode
    ai_disclosure: AiDisclosure
    recording: Literal[False]
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def _roles(self) -> Self:
        _both_roles(self.languages)
        return self


class CallTokenRequest(Contract):
    role: Role
    ai_disclosure_acknowledged: Literal[True]


class CallToken(Contract):
    token: str
    livekit_url: str | None = None  # None in demo mode: there is no live room to join
    identity: str
    expires_at: AwareDatetime


class CallRoomMetadata(Contract):
    session_id: UUID
    languages: dict[Role, LanguageTag]
    mode: CallMode
    glossary_id: str

    @model_validator(mode="after")
    def _roles(self) -> Self:
        _both_roles(self.languages)
        return self


class CallContext(Contract):
    """What the dentist's panel shows. It never holds an image."""

    session_id: UUID
    booking_id: UUID
    slot: Slot
    patient_locale: Locale
    analysis: AnalysisResult | None = None


class CaptionEvent(Contract):
    utterance_id: str = Field(min_length=1)
    speaker: Role
    src_lang: LanguageTag
    tgt_lang: LanguageTag
    src_text: str
    tgt_text: str  # empty while the event is not final, and when the translation failed
    final: bool
    translation_status: TranslationStatus = "ok"
    glossary_missing: tuple[str, ...] = ()
    digits_ok: bool
    audio: AudioMode
    t_speech_end_ms: int = Field(ge=0)
    t_first_audio_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _failure_is_final_and_silent(self) -> Self:
        if self.translation_status == "failed" and (
            self.tgt_text or not self.final or self.audio != "captions_only"
        ):
            raise ValueError("a failed translation is final, silent and has no target text")
        return self


class ReplayTrack(Contract):
    role: Role
    lang: LanguageTag
    url: str


class ReplayManifest(Contract):
    replay_id: str
    tracks: tuple[ReplayTrack, ...]
    events: tuple[CaptionEvent, ...]
    latency_p50_ms: int = Field(ge=0)
    latency_p95_ms: int = Field(ge=0)
    n: int = Field(ge=0)

    @model_validator(mode="after")
    def _percentiles(self) -> Self:
        if self.latency_p95_ms < self.latency_p50_ms:
            raise ValueError("p95 cannot be below p50")
        return self


class ChatEvent(Contract):
    """One translated chat message. The source text is always kept."""

    message_id: UUID
    sender: Role
    src_lang: LanguageTag
    tgt_lang: LanguageTag
    src_text: str = Field(min_length=1, max_length=1000)
    tgt_text: str | None = None
    translation_status: TranslationStatus
    glossary_missing: tuple[str, ...] = ()
    digits_ok: bool
    sent_at: AwareDatetime

    @model_validator(mode="after")
    def _text_matches_status(self) -> Self:
        if (self.translation_status == "ok") != (self.tgt_text is not None):
            raise ValueError("tgt_text is present exactly when the translation succeeded")
        return self
