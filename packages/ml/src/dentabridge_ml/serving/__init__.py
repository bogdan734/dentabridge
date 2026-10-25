"""Serving: one function turns a spec string into the VisionBackend the API and evals use."""

import os

from dentabridge_core import VisionBackend
from dentabridge_ml.serving.parse import ReleaseConfig

OLLAMA_URL = "http://localhost:11434"


def load_vision_backend(
    spec: str, release: ReleaseConfig, image_max_px: int | None = None
) -> VisionBackend:
    """`ollama:<model>` (server in DENTABRIDGE_OLLAMA_URL) or `transformers:<directory>`.

    `image_max_px` is the picture size the model was trained on; Transformers needs it.
    """
    kind, _, target = spec.partition(":")
    if kind == "ollama" and target:
        from dentabridge_ml.serving.ollama import OllamaVisionBackend

        url = os.environ.get("DENTABRIDGE_OLLAMA_URL", OLLAMA_URL)
        return OllamaVisionBackend(url, target, release, image_max_px=image_max_px)
    if kind == "transformers" and target:
        if image_max_px is None:
            raise ValueError("transformers backends need image_max_px, the size used in training")
        from dentabridge_ml.serving.transformers_backend import TransformersVisionBackend

        return TransformersVisionBackend(target, release, image_max_px)
    raise ValueError(f"unknown backend {spec!r}: use ollama:<model> or transformers:<directory>")
