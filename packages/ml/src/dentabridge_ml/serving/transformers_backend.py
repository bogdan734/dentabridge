"""VisionBackend over Transformers, greedy decoding.

It is the reference the Ollama round-trip is compared with, and the evaluation and serving route
where Ollama cannot load the tuned weights or the host (a rented GPU) has none. Needs the `train`
extra.
"""

import asyncio
import io
import time
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    Qwen3VLForConditionalGeneration,
)

from dentabridge_core import ASSESSABLE, Modality, ModelInfo, SanitisedImage, VisionOutput
from dentabridge_ml.serving.ollama import MAX_NEW_TOKENS
from dentabridge_ml.serving.parse import ParseError, ReleaseConfig, answer_codes, parse_output
from dentabridge_ml.serving.scores import scores_by_code, status_scores
from dentabridge_ml.targets.prompt import build_prompt, fit_image


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def dtype_for(device: str, capability: tuple[int, int] | None) -> torch.dtype:
    """bf16 where the hardware runs it natively (Ampere and later, Apple, CPU), else fp16."""
    if device == "cuda" and capability is not None and capability[0] < 8:
        return torch.float16
    return torch.bfloat16


class _StatusProbs(LogitsProcessor):  # type: ignore[misc]  # transformers is Any for mypy
    """Records, at every step and for every row, the probability of the two status tokens."""

    def __init__(self, token_ids: tuple[int, int]) -> None:
        self._ids = list(token_ids)
        self.steps: list[torch.Tensor] = []

    def __call__(self, input_ids: Any, scores: torch.Tensor) -> torch.Tensor:
        self.steps.append(torch.softmax(scores.float(), dim=-1)[:, self._ids].cpu())
        return scores  # observed, never changed


class TransformersVisionBackend:
    def __init__(
        self,
        model_dir: str,
        release: ReleaseConfig,
        image_max_px: int,
        dtype: torch.dtype | None = None,
        device: str | None = None,
    ) -> None:
        self.model = ModelInfo(model_id=model_dir, version="local", backend="transformers")
        self._release = release
        self._image_max_px = image_max_px
        self._device = device or pick_device()
        if dtype is None:
            capability = torch.cuda.get_device_capability() if self._device == "cuda" else None
            dtype = dtype_for(self._device, capability)
        self._processor: Any = AutoProcessor.from_pretrained(model_dir)
        self._processor.tokenizer.padding_side = "left"  # generation continues from the right edge
        self._net: Any = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype)
        self._net.to(self._device).eval()
        convert = self._processor.tokenizer.convert_tokens_to_ids
        self._ids: tuple[int, int] = (convert("possible"), convert("not"))

    def _generate(
        self, images: Sequence[SanitisedImage], prompt: str
    ) -> list[tuple[str, list[float] | None]]:
        """Per picture: the raw answer and one score per status value (None if not pairable)."""
        pictures = []
        for image in images:
            with Image.open(io.BytesIO(image.data)) as opened:
                pictures.append(fit_image(opened.convert("RGB"), self._image_max_px))
        turn = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(turn, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text] * len(pictures), images=pictures, padding=True, return_tensors="pt"
        ).to(self._device)
        probe = _StatusProbs(self._ids)
        with torch.no_grad():
            out = self._net.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,  # the checkpoint's sampling settings do not apply to greedy
                top_p=None,
                top_k=None,
                logits_processor=LogitsProcessorList([probe]),
            )
        tokenizer = self._processor.tokenizer
        probs = torch.stack(probe.steps)  # steps x rows x (possible, not)
        answers = []
        for row, ids in enumerate(out[:, inputs["input_ids"].shape[1] :].tolist()):
            raw = str(tokenizer.decode(ids, skip_special_tokens=True))
            tokens = [tokenizer.decode([i]) for i in ids]
            try:
                scores = status_scores(tokens, probs[:, row, 0].tolist(), probs[:, row, 1].tolist())
            except ValueError:
                scores = None
            answers.append((raw, scores))
        return answers

    async def analyse_many(
        self, images: Sequence[SanitisedImage], modality: Modality
    ) -> list[VisionOutput | ParseError]:
        """One forward pass for several pictures; latency is the batch time shared out evenly."""
        prompt = build_prompt(self._release.assessed & ASSESSABLE[modality], modality)
        started = time.perf_counter()
        generated = await asyncio.to_thread(self._generate, images, prompt)
        each_ms = (time.perf_counter() - started) * 1000 / len(images)
        results: list[VisionOutput | ParseError] = []
        for raw, scores in generated:
            try:
                paired = None if scores is None else scores_by_code(answer_codes(raw), scores)
                results.append(parse_output(raw, paired, self._release, latency_ms=each_ms))
            except ParseError as error:
                results.append(error)
        return results

    async def analyse(self, image: SanitisedImage, modality: Modality) -> VisionOutput:
        (result,) = await self.analyse_many([image], modality)
        if isinstance(result, ParseError):
            raise result
        return result
