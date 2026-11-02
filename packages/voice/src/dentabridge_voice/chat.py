"""Chat translation: one message in, one ChatEvent out, the source text always kept."""

import logging
from uuid import uuid4

from dentabridge_core import CallRoomMetadata, ChatEvent, Role, Translator
from dentabridge_voice.checks import digits_preserved, matched_entries, missing_terms
from dentabridge_voice.glossary import Glossary
from dentabridge_voice.ports import WallClock
from dentabridge_voice.routing import ParticipantInfo, direction_of

MAX_CHARS = 1000
log = logging.getLogger("dentabridge_voice")


class ChatTranslator:
    def __init__(self, translator: Translator, glossary: Glossary, clock: WallClock) -> None:
        self._translator = translator
        self._glossary = glossary
        self._clock = clock

    async def handle(self, raw: str, sender: Role, src: str, tgt: str) -> ChatEvent:
        """Translate one message. A translator that fails gives a "failed" event that still
        carries the source text, so nothing the person typed is lost."""
        text = raw.strip()
        if not text or len(text) > MAX_CHARS:
            raise ValueError(f"a chat message is 1 to {MAX_CHARS} characters")
        entries = self._glossary.for_direction(src, tgt)
        event = {
            "message_id": uuid4(),
            "sender": sender,
            "src_lang": src,
            "tgt_lang": tgt,
            "src_text": text,
            "sent_at": self._clock.now(),
        }
        try:
            translation = await self._translator.translate(
                text, src, tgt, matched_entries(text, entries)
            )
        except Exception as error:  # a separate model service: any failure becomes "failed"
            log.warning("chat translation failed: %s", type(error).__name__)
            return ChatEvent.model_validate(
                {**event, "tgt_text": None, "translation_status": "failed", "digits_ok": True}
            )
        return ChatEvent.model_validate(
            {
                **event,
                "tgt_text": translation.text,
                "translation_status": "ok",
                "glossary_missing": missing_terms(text, translation.text, entries),
                "digits_ok": digits_preserved(text, translation.text),
            }
        )


async def relay_chat(
    raw: str, sender: ParticipantInfo, meta: CallRoomMetadata, chat: ChatTranslator
) -> ChatEvent | None:
    """A message from the `chat.in.v1` stream, translated for `chat.v1`. None when the sender is
    not one of the call's two people (the interpreter itself included) or the message is empty or
    too long: a stream from the room is not ours to fail on."""
    direction = direction_of(sender, meta)
    if direction is None:
        return None
    try:
        return await chat.handle(raw, direction.role, direction.src, direction.tgt)
    except ValueError:
        log.warning("a chat message from the %s was empty or too long", direction.role)
        return None
