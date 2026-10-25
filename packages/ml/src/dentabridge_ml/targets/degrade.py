"""Synthetic "unusable photo" negatives, so the model can say "retake the photo".

Each kind maps to one QualityReason of the contract. Everything is seeded: the same image and
seed always give the same pixels.
"""

import random
from collections.abc import Mapping
from typing import Literal

from PIL import Image, ImageEnhance, ImageFilter

from dentabridge_core.analysis import QualityReason

DegradeKind = Literal["blur", "dark", "overexposed", "crop_no_teeth"]

DEGRADE_KINDS: tuple[DegradeKind, ...] = ("blur", "dark", "overexposed", "crop_no_teeth")
# An over-exposed photo has lost its detail, which is what "blurry" tells the patient to fix.
REASON: Mapping[DegradeKind, QualityReason] = {
    "blur": "blurry",
    "dark": "dark",
    "overexposed": "blurry",
    "crop_no_teeth": "no_teeth",
}


def degrade(img: Image.Image, kind: DegradeKind, seed: int) -> Image.Image:
    rng = random.Random(seed)
    rgb = img.convert("RGB")
    width, height = rgb.size
    if kind == "blur":
        return rgb.filter(ImageFilter.GaussianBlur(max(width, height) * rng.uniform(0.02, 0.05)))
    if kind == "dark":
        return ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.03, 0.12))
    if kind == "overexposed":
        bright = ImageEnhance.Brightness(rgb).enhance(rng.uniform(3.5, 6.0))
        return ImageEnhance.Contrast(bright).enhance(rng.uniform(0.3, 0.6))
    # crop_no_teeth: a small patch from a corner, enlarged: soft background, no teeth in focus
    side = rng.uniform(0.25, 0.32)
    box_w, box_h = int(width * side), int(height * side)
    left = 0 if rng.random() < 0.5 else width - box_w
    top = 0 if rng.random() < 0.5 else height - box_h
    patch = rgb.crop((left, top, left + box_w, top + box_h))
    return patch.resize((width, height), Image.Resampling.BICUBIC).filter(
        ImageFilter.GaussianBlur(max(width, height) * 0.01)
    )
