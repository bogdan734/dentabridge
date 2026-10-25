"""SFT rows and the batch collator.

A row is plain JSON (image path, prompt, answer, optional degradation), so a run can be audited
and resumed. Degraded "unusable photo" negatives are rebuilt on the fly from the seed: nothing
extra is written to disk. The loss covers the answer only, never the prompt or the image tokens.
"""

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict

from dentabridge_core import ASSESSABLE, FindingCode
from dentabridge_ml.data.records import ImageRecord
from dentabridge_ml.data.sources import SOURCES
from dentabridge_ml.splits.manifest import SplitManifest, SplitName
from dentabridge_ml.targets.degrade import DEGRADE_KINDS, REASON, DegradeKind, degrade
from dentabridge_ml.targets.prompt import build_prompt
from dentabridge_ml.targets.sft import target_json, unusable_json

END_OF_TURN = "<|im_end|>\n"


class Degradation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: DegradeKind
    seed: int


class SftRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    image: str  # relative to the data root: "<source>/<path>"
    prompt: str
    answer: str
    degrade: Degradation | None = None


def _asked(record: ImageRecord, rng: random.Random, subsets: bool) -> frozenset[FindingCode]:
    codes = sorted(
        SOURCES[record.source].coverage & ASSESSABLE["intraoral_photo"], key=list(FindingCode).index
    )
    if subsets and len(codes) > 1 and rng.random() < 0.5:
        # teach "judge only these" so the full five-code prompt used at inference is not new
        return frozenset(rng.sample(codes, rng.randint(1, len(codes))))
    return frozenset(codes)


def build_rows(
    records: Mapping[str, ImageRecord],
    manifest: SplitManifest,
    split: SplitName,
    negatives_frac: float,
    seed: int,
    subsets: bool = True,
) -> list[SftRow]:
    rng = random.Random(seed)
    chosen = [records[e.record_id] for e in manifest.of(split)]
    rows: list[SftRow] = []
    for record in chosen:
        asked = _asked(record, rng, subsets)
        rows.append(
            SftRow(
                id=record.record_id,
                image=f"{record.source}/{record.path}",
                prompt=build_prompt(asked, "intraoral_photo"),
                answer=target_json(record, asked),
            )
        )
    for k in range(round(len(rows) * negatives_frac)):
        record = rng.choice(chosen)
        kind = DEGRADE_KINDS[k % len(DEGRADE_KINDS)]
        asked = _asked(record, rng, subsets)
        rows.append(
            SftRow(
                id=f"{record.record_id}#{kind}{k}",
                image=f"{record.source}/{record.path}",
                prompt=build_prompt(asked, "intraoral_photo"),
                answer=unusable_json(REASON[kind]),
                degrade=Degradation(kind=kind, seed=rng.randrange(2**31)),
            )
        )
    rng.shuffle(rows)
    return rows


def write_sft(rows: Sequence[SftRow], out: Path) -> int:
    out.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def read_sft(path: Path) -> list[SftRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [SftRow.model_validate(json.loads(line)) for line in lines if line.strip()]


class SftCollator:
    """Turns rows into model inputs. `processor` is a Qwen3-VL processor (tokenizer + images)."""

    def __init__(self, processor: Any, root: Path, image_max_px: int, max_seq_len: int) -> None:
        self._processor = processor
        self._root = root
        self._image_max_px = image_max_px
        self._max_seq_len = max_seq_len

    def _image(self, row: SftRow) -> Image.Image:
        with Image.open(self._root / row.image) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if row.degrade is not None:
            image = degrade(image, row.degrade.kind, row.degrade.seed)
        image.thumbnail((self._image_max_px, self._image_max_px), Image.Resampling.LANCZOS)
        return image

    def __call__(self, rows: Sequence[SftRow]) -> dict[str, Any]:
        images = [self._image(r) for r in rows]
        turns = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r.prompt}]}]
            for r in rows
        ]
        prompts = [
            self._processor.apply_chat_template(t, tokenize=False, add_generation_prompt=True)
            for t in turns
        ]
        texts = [p + r.answer + END_OF_TURN for p, r in zip(prompts, rows, strict=True)]
        batch = self._processor(text=texts, images=images, padding=True, return_tensors="pt")
        prompt_batch = self._processor(
            text=prompts, images=images, padding=True, return_tensors="pt"
        )
        if batch["input_ids"].shape[1] > self._max_seq_len:
            raise ValueError(
                f"a row needs {batch['input_ids'].shape[1]} tokens, limit {self._max_seq_len}"
            )
        labels = batch["input_ids"].clone()
        for i, length in enumerate(prompt_batch["attention_mask"].sum(dim=1).tolist()):
            labels[i, :length] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return dict(batch)
