"""Builders, fakes and HTTP flow steps for the API's tests, and for the fixture recorder."""

import io
from collections.abc import Iterator
from typing import Any

import httpx
import numpy as np
from PIL import Image

from dentabridge_core import (
    ErrorCode,
    ErrorResponse,
    Finding,
    ModelInfo,
    SanitisedImage,
    VisionOutput,
)

CLINIC = "alfama-demo-dental"
_keys = iter(range(1, 10_000))


def jpeg_bytes(
    size: tuple[int, int] = (120, 80), noise: bool = False, exif: Image.Exif | None = None
) -> bytes:
    if noise:  # incompressible: a big file with few pixels
        pixels = np.random.default_rng(1).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
    else:
        image = Image.new("RGB", size, (200, 120, 110))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95, **({"exif": exif} if exif is not None else {}))
    return buffer.getvalue()


class ScriptedVision:
    """Answers the same observations for every picture."""

    model = ModelInfo(model_id="scripted", version="1", backend="fixture")

    def __init__(self, findings: tuple[Finding, ...] = (), note: str | None = None) -> None:
        self.findings = findings
        self.note = note
        self.calls = 0

    async def analyse(self, image: SanitisedImage, modality: str) -> VisionOutput:
        self.calls += 1
        return VisionOutput(
            image_quality="ok", findings=self.findings, note_en=self.note, latency_ms=3.0
        )


def key() -> dict[str, str]:
    return {"Idempotency-Key": f"booking-{next(_keys):05d}"}


def code_of(response: httpx.Response) -> ErrorCode:
    return ErrorResponse.model_validate(response.json()).code


def quote(client: Any, **changes: object) -> dict[str, Any]:
    body = {
        "clinic_id": CLINIC,
        "treatment_codes": ["crown"],
        "origin_iata": "LHR",
        "depart_date": "2026-11-10",
        "return_date": "2026-11-20",
        "travellers": 1,
        "currency": "EUR",
        **changes,
    }
    response = client.post("/v1/quotes", json=body)
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


def first_slot(
    client: Any, clinic: str = CLINIC, start: str = "2026-11-11", end: str = "2026-11-14"
) -> dict[str, Any]:
    slots = client.get(f"/v1/clinics/{clinic}/slots", params={"from": start, "to": end}).json()
    assert slots, "the seed schedule has slots in the window"
    result: dict[str, Any] = slots[0]
    return result


def hold(client: Any, slot_id: str) -> httpx.Response:
    response: httpx.Response = client.post("/v1/holds", json={"slot_id": slot_id}, headers=key())
    return response


def book(client: Any, hold_id: str, quote_id: str, **extra: str) -> httpx.Response:
    response: httpx.Response = client.post(
        "/v1/bookings", json={"hold_id": hold_id, "quote_id": quote_id, **extra}, headers=key()
    )
    return response


def consent(client: Any, purpose: str) -> str:
    response = client.post(
        "/v1/consents",
        json={
            "purpose": purpose,
            "health_data_explicit": True,
            "text_version": "v1",
            "locale": "en",
        },
    )
    return str(response.json()["consent_id"])


def an_analysis(client: Any, sample_id: str = "one") -> str:
    response = client.post(
        "/v1/analyses/sample", json={"sample_id": sample_id, "locale": "en"}, headers=key()
    )
    return str(response.json()["analysis_id"])


def field_names(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for name, item in value.items():
            yield name
            yield from field_names(item)
    elif isinstance(value, list):
        for item in value:
            yield from field_names(item)
