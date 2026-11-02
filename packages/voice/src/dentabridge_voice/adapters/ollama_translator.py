"""A Translator over a local Ollama server.

The model is whatever `model` names (a permissive-licence instruct model by default; the V4 bake-off
decides). The prompt carries only the glossary terms the sentence uses, so it stays short and the
model is not tempted to bring in terms nobody said.
"""

import time
from collections.abc import Sequence

import httpx

from dentabridge_core import GlossaryEntry, Translation

LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "tr": "Turkish",
    "th": "Thai",
}
TIMEOUT_S = 30.0


def _term(side: str) -> str:
    return side.split("|")[0]  # the first alternative is the full dictionary form


class OllamaTranslator:
    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient | None = None) -> None:
        self.model_id = model
        self._url = base_url.rstrip("/") + "/api/chat"
        self._client = client or httpx.AsyncClient(timeout=TIMEOUT_S)

    @staticmethod
    def _system(src: str, tgt: str, glossary: Sequence[GlossaryEntry]) -> str:
        prompt = (
            "You are the interpreter in a dental consultation between a patient and a dentist. "
            f"Translate the message from {LANGUAGE_NAMES.get(src, src)} to "
            f"{LANGUAGE_NAMES.get(tgt, tgt)}. Reply with only the translation, nothing else. "
            "Keep every number, tooth number, dose and unit exactly as written. Do not add, leave "
            "out or soften anything. If the message is a question, translate it, do not answer it."
        )
        if glossary:
            prompt += " Use these terms: " + "; ".join(
                f"{_term(e.src)} = {_term(e.tgt)}" for e in glossary
            )
        return prompt

    async def translate(
        self, text: str, src: str, tgt: str, glossary: Sequence[GlossaryEntry]
    ) -> Translation:
        started = time.perf_counter()
        response = await self._client.post(
            self._url,
            json={
                "model": self.model_id,
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": self._system(src, tgt, glossary)},
                    {"role": "user", "content": text},
                ],
            },
        )
        response.raise_for_status()
        translated = str(response.json()["message"]["content"]).strip()
        if not translated:
            raise ValueError("the model returned nothing")
        return Translation(text=translated, latency_ms=(time.perf_counter() - started) * 1000)
