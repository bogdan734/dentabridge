"""What the pipeline needs from the outside: speech in, speech out, and a clock in milliseconds."""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from dentabridge_core import Role


class Utterance(BaseModel):
    """One segment of one person's speech: 16-bit mono PCM and when they stopped talking."""

    model_config = ConfigDict(frozen=True)

    utterance_id: str = Field(min_length=1)
    speaker: Role
    pcm: bytes
    sample_rate: int = Field(gt=0)
    t_speech_end_ms: int = Field(ge=0)


class AudioChunk(BaseModel):
    """A piece of translated speech, 16-bit mono PCM, stamped when it was produced."""

    model_config = ConfigDict(frozen=True)

    utterance_id: str
    pcm: bytes
    sample_rate: int
    t_ms: int


@runtime_checkable
class SpeechToText(Protocol):
    async def transcribe(self, pcm: bytes, sample_rate: int, lang: str) -> str: ...


@runtime_checkable
class TextToSpeech(Protocol):
    sample_rate: int

    def synthesise(self, text: str, lang: str) -> AsyncIterator[bytes]: ...


class Clock(Protocol):
    """Milliseconds on the call's timeline, the one `Utterance.t_speech_end_ms` is read on."""

    def now_ms(self) -> int: ...


class WallClock(Protocol):
    """Calendar time, aware (chat messages carry the moment they were sent)."""

    def now(self) -> datetime: ...
