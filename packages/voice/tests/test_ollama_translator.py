import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from dentabridge_core import GlossaryEntry, Translator
from dentabridge_voice.adapters.ollama_translator import OllamaTranslator

TOOTH = GlossaryEntry(src="tooth|teeth", tgt="зуб|зуб*")
PAIN = GlossaryEntry(src="pain|ache", tgt="боль|бол*")


def translator(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaTranslator:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaTranslator("http://ollama.test", "qwen3:4b-instruct", client=client)


def reply(text: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})

    return handler


def translate(t: OllamaTranslator, text: str, glossary: list[GlossaryEntry]) -> Any:
    return asyncio.run(t.translate(text, "en", "ru", glossary))


def test_translator_mock_transport() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return reply("  мой зуб болит \n")(request)

    t = translator(handler)
    assert isinstance(t, Translator)
    result = translate(t, "my tooth hurts", [TOOTH])
    assert (result.text, result.latency_ms >= 0) == ("мой зуб болит", True)
    (request,) = seen
    body = json.loads(request.content)
    assert (request.method, request.url.path) == ("POST", "/api/chat")
    assert (body["model"], body["stream"], body["think"]) == ("qwen3:4b-instruct", False, False)
    assert body["options"]["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "my tooth hurts"
    assert t.model_id == "qwen3:4b-instruct"


def test_the_prompt_names_the_languages_and_the_safety_rules() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"][0]["content"])
        return reply("да")(request)

    translate(translator(handler), "yes", [])
    system = seen[0]
    assert "from English to Russian" in system
    assert "only the translation" in system
    assert "Keep every number" in system
    assert "Use these terms" not in system, "no glossary line when nothing matched"


def test_prompt_contains_only_matched_terms() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"][0]["content"])
        return reply("мой зуб болит")(request)

    translate(translator(handler), "my tooth", [TOOTH])
    translate(translator(handler), "my tooth and pain", [TOOTH, PAIN])
    assert "tooth = зуб" in seen[0]
    assert "pain" not in seen[0]
    assert "tooth = зуб; pain = боль" in seen[1]
    assert "зуб*" not in seen[1], "the prompt quotes the full form, not the matching pattern"


def test_a_server_error_or_an_empty_answer_is_an_error() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        translate(translator(lambda r: httpx.Response(500, json={"error": "boom"})), "hi", [])
    with pytest.raises(ValueError, match="nothing"):
        translate(translator(reply("   ")), "hi", [])
