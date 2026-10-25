"""The bundled sample pictures: what the public demo can analyse, with the credit they require."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from dentabridge_api.uploads import sanitise_upload
from dentabridge_core import Modality, SanitisedImage

CATALOGUE = "samples.json"


class SampleEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    file: str
    modality: Modality
    attribution: str  # author, title, source, as the licence asks
    licence: str  # for example "CC BY 4.0"
    directory: Path

    def sanitised(self) -> SanitisedImage:
        return sanitise_upload((self.directory / self.file).read_bytes(), "image/jpeg")

    def jpeg(self) -> bytes:
        return (self.directory / self.file).read_bytes()


class SampleCatalogue:
    def __init__(self, directory: Path) -> None:
        raw = json.loads((directory / CATALOGUE).read_text(encoding="utf-8"))
        self._entries = {e["id"]: SampleEntry(directory=directory, **e) for e in raw}

    def all(self) -> list[SampleEntry]:
        return list(self._entries.values())

    def get(self, sample_id: str) -> SampleEntry | None:
        return self._entries.get(sample_id)
