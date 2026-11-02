import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from dentabridge_core import ChatEvent, GlossaryEntry, Translation
from dentabridge_voice.chat import MAX_CHARS, ChatTranslator
from dentabridge_voice.glossary import Glossary

NOW = datetime(2026, 11, 2, 12, tzinfo=UTC)
GLOSSARY = Glossary(langs=("en", "ru"), entries=(GlossaryEntry(src="tooth|teeth", tgt="зуб*"),))


class FixedClock:
    def now(self) -> datetime:
        return NOW


class Translating:
    model_id = "fake"

    def __init__(self, text: str | Exception) -> None:
        self.text = text
        self.glossaries: list[Sequence[GlossaryEntry]] = []

    async def translate(
        self, text: str, src: str, tgt: str, glossary: Sequence[GlossaryEntry]
    ) -> Translation:
        self.glossaries.append(glossary)
        if isinstance(self.text, Exception):
            raise self.text
        return Translation(text=self.text, latency_ms=1.0)


def chat(raw: str, translator: Translating) -> ChatEvent:
    handler = ChatTranslator(translator, GLOSSARY, FixedClock())
    return asyncio.run(handler.handle(raw, "patient", "en", "ru"))


def test_chat_event_matches_core_schema() -> None:
    event = chat("my tooth hurts", Translating("мой зуб болит"))
    assert ChatEvent.model_validate(event.model_dump()) == event
    assert (event.sender, event.src_lang, event.tgt_lang) == ("patient", "en", "ru")
    assert (event.src_text, event.tgt_text, event.translation_status) == (
        "my tooth hurts",
        "мой зуб болит",
        "ok",
    )
    assert (event.sent_at, event.glossary_missing, event.digits_ok) == (NOW, (), True)


def test_digit_swap_flagged_in_chat() -> None:
    event = chat("tooth 36 hurts", Translating("зуб 63 болит"))
    assert event.digits_ok is False


def test_a_missing_term_is_reported_in_chat() -> None:
    assert chat("my tooth", Translating("моя проблема")).glossary_missing == ("tooth",)


def test_translator_failure_keeps_source() -> None:
    event = chat("my tooth hurts", Translating(RuntimeError("model unavailable")))
    assert (event.translation_status, event.tgt_text, event.src_text) == (
        "failed",
        None,
        "my tooth hurts",
    )


def test_the_translator_gets_only_the_terms_the_message_uses() -> None:
    translator = Translating("да")
    chat("yes", translator)
    assert translator.glossaries == [()]


@pytest.mark.parametrize("raw", ["", "   \n", "x" * (MAX_CHARS + 1)])
def test_message_length_capped(raw: str) -> None:
    with pytest.raises(ValueError, match="characters"):
        chat(raw, Translating("never reached"))


def test_a_message_of_exactly_the_cap_is_translated() -> None:
    assert chat("x" * MAX_CHARS, Translating("да")).translation_status == "ok"


def test_every_message_gets_its_own_id() -> None:
    translator: Any = Translating("да")
    assert chat("a", translator).message_id != chat("a", translator).message_id
