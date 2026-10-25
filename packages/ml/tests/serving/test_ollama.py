import asyncio
import base64
import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image

from dentabridge_core import FindingCode, SanitisedImage
from dentabridge_ml.serving.ollama import OllamaVisionBackend, answer_schema
from dentabridge_ml.serving.parse import ParseError, ReleaseConfig
from dentabridge_ml.targets.prompt import build_prompt

CARIES, FILLING = FindingCode.CARIES, FindingCode.FILLING
IMAGE = SanitisedImage(data=b"\xff\xd8jpeg", width=8, height=8, sha256="a" * 64)
RELEASE = ReleaseConfig(assessed=frozenset({CARIES, FILLING, FindingCode.CALCULUS}))
ANSWER = {
    "image_quality": "ok",
    "quality_reason": None,
    "findings": [{"code": "caries", "status": "possible", "regions": []}],
}


def _backend(
    content: str, seen: list[dict[str, Any]], status: int = 200, image_max_px: int | None = None
) -> OllamaVisionBackend:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaVisionBackend(
        "http://ollama.test:11434/", "qwen3-vl:4b", RELEASE, client, image_max_px=image_max_px
    )


def test_ollama_request_shape() -> None:
    seen: list[dict[str, Any]] = []
    output = asyncio.run(_backend(json.dumps(ANSWER), seen).analyse(IMAGE, "intraoral_photo"))
    (body,) = seen
    assert body["model"] == "qwen3-vl:4b"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0, "num_predict": 700}
    (message,) = body["messages"]
    assert message["images"] == [base64.b64encode(IMAGE.data).decode()]
    asked = frozenset({CARIES, FILLING})  # calculus is never assessable on a photo
    assert message["content"] == build_prompt(asked, "intraoral_photo")
    assert body["format"] == answer_schema(asked)
    assert output.findings[0].status == "possible"
    assert output.latency_ms >= 0


def test_the_schema_lists_only_the_asked_codes_in_catalogue_order() -> None:
    schema = answer_schema(frozenset({FindingCode.GINGIVITIS, CARIES}))
    codes = schema["properties"]["findings"]["items"]["properties"]["code"]
    assert codes == {"enum": ["caries", "gingivitis"]}
    assert schema["required"] == ["image_quality", "quality_reason", "findings"]


def test_a_bad_answer_is_a_parse_error_and_a_server_error_is_not_hidden() -> None:
    with pytest.raises(ParseError):
        asyncio.run(_backend("I see a cavity.", []).analyse(IMAGE, "intraoral_photo"))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_backend("", [], status=500).analyse(IMAGE, "intraoral_photo"))


def test_backend_reports_its_identity() -> None:
    backend = _backend("{}", [])
    assert (backend.model.model_id, backend.model.backend) == ("qwen3-vl:4b", "ollama")


def test_the_picture_is_fitted_to_the_size_the_model_was_trained_on() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), (90, 20, 20)).save(buffer, "JPEG")
    big = SanitisedImage(data=buffer.getvalue(), width=200, height=100, sha256="b" * 64)
    seen: list[dict[str, Any]] = []
    asyncio.run(_backend(json.dumps(ANSWER), seen, image_max_px=64).analyse(big, "intraoral_photo"))
    sent = base64.b64decode(seen[0]["messages"][0]["images"][0])
    with Image.open(io.BytesIO(sent)) as image:
        assert image.size == (64, 32)
