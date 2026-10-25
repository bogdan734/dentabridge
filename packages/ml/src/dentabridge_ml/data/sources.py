"""Where each training source comes from, what its labels cover and what its licence allows.

Sizes and checksums come from the publishers' own metadata (Zenodo API, Mendeley Data API),
read on 2026-10-25. A published source may feed released weights; a private one never does.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from dentabridge_core import FindingCode

SourceId = Literal["aku_caries", "alphadent", "mendeley_gingivitis", "kaggle_oral_private"]
Camera = Literal["phone", "clinic", "xray"]


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    size: int
    md5: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class SourceSpec:
    title: str
    licence: str
    page: str
    files: tuple[RemoteFile, ...]
    published: bool
    # Codes this source labels. A code outside it is "not labelled here", never "not seen".
    coverage: frozenset[FindingCode]
    camera: Camera


SOURCES: Mapping[SourceId, SourceSpec] = MappingProxyType(
    {
        "aku_caries": SourceSpec(
            title="Annotated intraoral image dataset for dental caries detection (AKU)",
            licence="CC-BY-4.0",
            page="https://zenodo.org/records/14827784",
            files=(
                RemoteFile(
                    name="Dataset.zip",
                    url="https://zenodo.org/api/records/14827784/files/Dataset.zip/content",
                    size=1_576_115_314,
                    md5="89307871ac2f08f4a8f3a7da1f18db31",
                ),
            ),
            published=True,
            coverage=frozenset({FindingCode.CARIES}),
            camera="phone",
        ),
        "alphadent": SourceSpec(
            title="AlphaDent: a dataset for automated tooth pathology detection",
            licence="CC-BY-4.0",
            page="https://zenodo.org/records/16582489",
            files=(
                RemoteFile(
                    name="AlphaDent.zip",
                    url="https://zenodo.org/api/records/16582489/files/AlphaDent.zip/content",
                    size=4_877_554_182,
                    md5="196953cd74e92240ddc77f433af80386",
                ),
            ),
            published=True,
            coverage=frozenset(
                {
                    FindingCode.CARIES,
                    FindingCode.FILLING,
                    FindingCode.CROWN,
                    FindingCode.ABRASION,
                }
            ),
            camera="clinic",
        ),
        "mendeley_gingivitis": SourceSpec(
            title="A dental intraoral image dataset of gingivitis for image captioning",
            licence="CC-BY-4.0",
            page="https://data.mendeley.com/datasets/3253gj88rr/1",
            files=(
                RemoteFile(
                    name="Dataset.rar",
                    url=(
                        "https://data.mendeley.com/public-files/datasets/3253gj88rr/files/"
                        "5349ba50-099f-4521-8e09-7e0ceb8a6b5e/file_downloaded"
                    ),
                    size=6_552_111_937,
                    sha256="16669f74665ff0deca6ddd25feb0d233e52fdc2313265465bf91731af1e8956b",
                ),
            ),
            published=True,
            coverage=frozenset({FindingCode.GINGIVITIS}),
            camera="clinic",
        ),
        # Licence "Unknown" on its Data Card and images taken from websites: local tests only.
        # It needs a Kaggle login, so there is nothing to download automatically.
        "kaggle_oral_private": SourceSpec(
            title="Kaggle Oral Diseases (private tests only)",
            licence="Unknown",
            page="https://www.kaggle.com/datasets/salmansajid05/oral-diseases",
            files=(),
            published=False,
            coverage=frozenset(
                {
                    FindingCode.CARIES,
                    FindingCode.CALCULUS,
                    FindingCode.GINGIVITIS,
                    FindingCode.DISCOLOURATION,
                    FindingCode.ULCER,
                    FindingCode.HYPODONTIA,
                }
            ),
            camera="clinic",
        ),
    }
)


def default_root() -> Path:
    """Where datasets live (gitignored). DENTABRIDGE_DATA_ROOT shares one copy across worktrees."""
    env = os.environ.get("DENTABRIDGE_DATA_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[5] / "data"
