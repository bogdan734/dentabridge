"""VisionBackend over Ollama's chat API: zero-shot baselines and, if ML1 passes, the tuned model.

Ollama's `format` parameter takes a JSON schema and constrains decoding to it. The schema lists
only the codes being asked about, so a baseline cannot answer about anything else.
"""

import base64
import io
import time
from typing import Any

import httpx
from PIL import Image

from dentabridge_core import (
    ASSESSABLE,
    FindingCode,
    Modality,
    ModelInfo,
    SanitisedImage,
    VisionOutput,
)
from dentabridge_ml.serving.parse import ReleaseConfig, parse_output
from dentabridge_ml.targets.prompt import build_prompt, fit_image

MAX_NEW_TOKENS = 700
_REGION = {
    "type": "object",
    "properties": {k: {"type": "number"} for k in ("x0", "y0", "x1", "y1")},
    "required": ["x0", "y0", "x1", "y1"],
}


def answer_schema(codes: frozenset[FindingCode]) -> dict[str, Any]:
    ordered = [c.value for c in FindingCode if c in codes]
    return {
        "type": "object",
        "properties": {
            "image_quality": {"enum": ["ok", "unusable"]},
            "quality_reason": {"enum": [None, "blurry", "dark", "no_teeth", "not_dental"]},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"enum": ordered},
                        "status": {"enum": ["possible", "not_seen"]},
                        "regions": {"type": "array", "items": _REGION},
                    },
                    "required": ["code", "status", "regions"],
                },
            },
        },
        "required": ["image_quality", "quality_reason", "findings"],
    }


class OllamaVisionBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        release: ReleaseConfig,
        client: httpx.AsyncClient | None = None,
        image_max_px: int | None = None,
        constrain: bool = True,
    ) -> None:
        self.model = ModelInfo(model_id=model, version="ollama", backend="ollama")
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._release = release
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=600.0))
        self._image_max_px = image_max_px  # the size the tuned model was trained on; None = as is
        self._constrain = constrain  # False: free decoding, to compare weights with other runtimes

    def _payload(self, image: SanitisedImage) -> str:
        data = image.data
        if self._image_max_px is not None:
            with Image.open(io.BytesIO(data)) as opened:
                fitted = fit_image(opened.convert("RGB"), self._image_max_px)
            buffer = io.BytesIO()
            fitted.save(buffer, "JPEG", quality=90)
            data = buffer.getvalue()
        return base64.b64encode(data).decode("ascii")

    async def analyse(self, image: SanitisedImage, modality: Modality) -> VisionOutput:
        asked = self._release.assessed & ASSESSABLE[modality]
        body: dict[str, Any] = {
            "model": self.model.model_id,
            "stream": False,
            "options": {"temperature": 0, "num_predict": MAX_NEW_TOKENS},
            "messages": [
                {
                    "role": "user",
                    "content": build_prompt(asked, modality),
                    "images": [self._payload(image)],
                }
            ],
        }
        if self._constrain:
            body["format"] = answer_schema(asked)
        started = time.perf_counter()
        response = await self._client.post(self._url, json=body)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - started) * 1000
        raw: str = response.json()["message"]["content"]
        return parse_output(raw, None, self._release, latency_ms=elapsed_ms)
