"""A VisionBackend for the demo and for CI: answers for the bundled sample pictures only.

`precompute_samples` runs the shipped model over the samples once and writes its answers next to
them; this backend replays them, so the public demo needs no model and gives the same answer
every time. A picture it does not know is refused: it would otherwise have to invent an answer.
"""

import json
from pathlib import Path

from pydantic import TypeAdapter

from dentabridge_api.samples import SampleCatalogue
from dentabridge_core import Modality, ModelInfo, SanitisedImage, VisionBackend, VisionOutput

PRECOMPUTED = "precomputed.json"
_ANSWERS = TypeAdapter(dict[str, VisionOutput])


class UnknownImage(RuntimeError):
    pass


class FixtureVisionBackend:
    def __init__(self, samples_dir: Path) -> None:
        document = json.loads((samples_dir / PRECOMPUTED).read_text(encoding="utf-8"))
        self.model = ModelInfo.model_validate(document["model"])
        self._answers = _ANSWERS.validate_python(document["answers"])

    async def analyse(self, image: SanitisedImage, modality: Modality) -> VisionOutput:
        try:
            return self._answers[image.sha256]
        except KeyError:
            raise UnknownImage("demo mode analyses the bundled samples only") from None


async def precompute_samples(backend: VisionBackend, samples_dir: Path) -> None:
    """Analyse every sample and write the answers the FixtureVisionBackend replays."""
    answers: dict[str, VisionOutput] = {}
    for sample in SampleCatalogue(samples_dir).all():
        image = sample.sanitised()
        answers[image.sha256] = await backend.analyse(image, sample.modality)
    document = {
        "model": backend.model.model_dump(mode="json"),
        "answers": _ANSWERS.dump_python(answers, mode="json"),
    }
    (samples_dir / PRECOMPUTED).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
