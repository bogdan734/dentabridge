"""One utterance in, captions and translated speech out.

The listener sees the words before they hear them: a caption with only the source text first, the
caption with the translation as soon as it exists, then the audio, then one more caption that
carries the time of the first audio. When the voice fails (or there is none) the caption says
"captions_only" instead of leaving the listener waiting for sound.
"""

from collections.abc import AsyncIterator

from dentabridge_core import CaptionEvent, Translator
from dentabridge_voice.checks import digits_preserved, matched_entries, missing_terms
from dentabridge_voice.glossary import Glossary
from dentabridge_voice.ports import AudioChunk, Clock, SpeechToText, TextToSpeech, Utterance


class Pipeline:
    def __init__(
        self,
        stt: SpeechToText,
        translator: Translator,
        tts: TextToSpeech | None,
        glossary: Glossary,
        clock: Clock,
    ) -> None:
        self._stt = stt
        self._translator = translator
        self._tts = tts
        self._glossary = glossary
        self._clock = clock

    async def handle(
        self, utt: Utterance, src: str, tgt: str
    ) -> AsyncIterator[CaptionEvent | AudioChunk]:
        """Speech in `src`, captions and speech in `tgt`. Silence yields nothing; a failing
        speech-to-text or translator raises (the worker decides what the room is told)."""
        heard = (await self._stt.transcribe(utt.pcm, utt.sample_rate, src)).strip()
        if not heard:
            return

        def caption(text: str, final: bool, **fields: object) -> CaptionEvent:
            return CaptionEvent.model_validate(
                {
                    "utterance_id": utt.utterance_id,
                    "speaker": utt.speaker,
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "src_text": heard,
                    "tgt_text": text,
                    "final": final,
                    "glossary_missing": (),
                    "digits_ok": True,
                    "audio": "spoken" if self._tts else "captions_only",
                    "t_speech_end_ms": utt.t_speech_end_ms,
                    **fields,
                }
            )

        yield caption("", final=False)
        entries = self._glossary.for_direction(src, tgt)
        translation = await self._translator.translate(
            heard, src, tgt, matched_entries(heard, entries)
        )
        checks = {
            "glossary_missing": missing_terms(heard, translation.text, entries),
            "digits_ok": digits_preserved(heard, translation.text),
        }
        yield caption(translation.text, final=True, **checks)
        if self._tts is None:
            return

        first_audio_ms: int | None = None
        try:
            async for pcm in self._tts.synthesise(translation.text, tgt):
                now = self._clock.now_ms()
                first_audio_ms = now if first_audio_ms is None else first_audio_ms
                yield AudioChunk(
                    utterance_id=utt.utterance_id,
                    pcm=pcm,
                    sample_rate=self._tts.sample_rate,
                    t_ms=now,
                )
        except Exception:  # the voice engine is a separate service; any failure ends in captions
            yield caption(translation.text, final=True, audio="captions_only", **checks)
        else:
            yield caption(translation.text, final=True, t_first_audio_ms=first_audio_ms, **checks)
