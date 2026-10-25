import pytest

from dentabridge_core import FindingCode
from dentabridge_ml.serving import load_vision_backend
from dentabridge_ml.serving.ollama import OllamaVisionBackend
from dentabridge_ml.serving.parse import ReleaseConfig

RELEASE = ReleaseConfig(assessed=frozenset({FindingCode.CARIES}))


def test_ollama_spec_names_the_model_after_the_colon() -> None:
    backend = load_vision_backend("ollama:qwen3-vl:4b", RELEASE, image_max_px=448)
    assert isinstance(backend, OllamaVisionBackend)
    assert backend.model.model_id == "qwen3-vl:4b"


@pytest.mark.parametrize("spec", ["", "ollama", "ollama:", "gpt:4", "transformers:"])
def test_unknown_or_empty_specs_are_refused(spec: str) -> None:
    with pytest.raises(ValueError, match="ollama:<model>"):
        load_vision_backend(spec, RELEASE)


def test_transformers_needs_the_training_picture_size() -> None:
    with pytest.raises(ValueError, match="image_max_px"):
        load_vision_backend("transformers:/some/dir", RELEASE)
