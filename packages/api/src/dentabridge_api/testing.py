"""Builders and fakes for the API's tests (and for other packages that test against it)."""

import io

import numpy as np
from PIL import Image

from dentabridge_core import Finding, ModelInfo, SanitisedImage, VisionOutput


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
