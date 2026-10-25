"""Pseudo-patients: photos of one mouth must never sit on both sides of a split.

Some sources give a real patient id. The others (AKU is split by image, with about ten photos per
child) get groups from near-duplicate hashes and embedding similarity, joined transitively.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from dentabridge_ml.data.records import ImageRecord

GroupMethod = Literal["patient_id", "cluster"]
BLOCK = 1024


def phash(path: Path) -> int:
    """64-bit difference hash: robust to rescaling and mild re-compression."""
    with Image.open(path) as image:
        small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class _Components:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[max(root_a, root_b)] = min(root_a, root_b)


def _close_pairs(
    emb: NDArray[np.floating], hashes: NDArray[np.uint64], max_cos_dist: float, max_hamming: int
) -> list[tuple[int, int]]:
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    pairs: list[tuple[int, int]] = []
    for start in range(0, len(unit), BLOCK):
        stop = min(start + BLOCK, len(unit))
        cosine_close = (1.0 - unit[start:stop] @ unit.T) <= max_cos_dist
        hash_close = np.bitwise_count(hashes[start:stop, None] ^ hashes[None, :]) <= max_hamming
        rows, cols = np.nonzero(cosine_close | hash_close)
        pairs += [(start + r, int(c)) for r, c in zip(rows, cols, strict=True) if start + r < c]
    return pairs


def group(
    records: Sequence[ImageRecord],
    emb: NDArray[np.floating],
    hashes: Sequence[int],
    max_cos_dist: float,
    max_hamming: int,
) -> list[str]:
    """One group id per record. A real patient id wins; the rest are clustered transitively."""
    if not (len(records) == len(emb) == len(hashes)):
        raise ValueError("records, embeddings and hashes must line up")
    ids = [r.record_id for r in records]
    free = [i for i, r in enumerate(records) if r.patient_id is None]
    result = [f"p:{r.source}:{r.patient_id}" if r.patient_id is not None else "" for r in records]
    if free:
        sub_hashes = np.array([hashes[i] for i in free], dtype=np.uint64)
        components = _Components(len(free))
        for a, b in _close_pairs(emb[free], sub_hashes, max_cos_dist, max_hamming):
            components.union(a, b)
        smallest: dict[int, str] = {}
        for local, i in enumerate(free):
            root = components.find(local)
            smallest[root] = min(smallest.get(root, ids[i]), ids[i])
        for local, i in enumerate(free):
            result[i] = f"c:{smallest[components.find(local)]}"
    return result


def group_method(record: ImageRecord) -> GroupMethod:
    return "patient_id" if record.patient_id is not None else "cluster"
