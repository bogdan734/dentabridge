"""One entry point to every source's records."""

from collections.abc import Callable
from pathlib import Path

from dentabridge_ml.data.aku import load_aku
from dentabridge_ml.data.alphadent import load_alphadent
from dentabridge_ml.data.records import ImageRecord
from dentabridge_ml.data.sources import SourceId

LOADERS: dict[SourceId, Callable[[Path], list[ImageRecord]]] = {
    "aku_caries": load_aku,
    "alphadent": load_alphadent,
}


def load_records(source: SourceId, root: Path) -> list[ImageRecord]:
    if source not in LOADERS:
        raise NotImplementedError(f"no loader for {source} yet")
    return LOADERS[source](root)
