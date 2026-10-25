"""VisionBackend over Transformers, greedy decoding.

It is the reference the Ollama round-trip is compared with, and the evaluation and serving route
where Ollama cannot load the tuned weights or the host (a rented GPU) has none. Needs the `train`
extra.
"""

import asyncio
import io
import time
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from dentabridge_core import ASSESSABLE, Modality, ModelInfo, SanitisedImage, VisionOutput
from dentabridge_ml.serving.ollama import MAX_NEW_TOKENS
from dentabridge_ml.serving.parse import ReleaseConfig, answer_codes, parse_output
from dentabridge_ml.serving.scores import scores_by_code, status_scores
from dentabridge_ml.targets.prompt import build_prompt, fit_image


def pick_device() -> str:
    if torch.cuda.is_available():  # ROCm builds report themselves as cuda
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class TransformersVisionBackend:
    def __init__(
        self,
        model_dir: str,
        release: ReleaseConfig,
        image_max_px: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
    ) -> None:
        self.model = ModelInfo(model_id=model_dir, version="local", backend="transformers")
        self._release = release
        self._image_max_px = image_max_px
        self._device = device or pick_device()
        self._processor: Any = AutoProcessor.from_pretrained(model_dir)
        self._net: Any = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype)
        self._net.to(self._device).eval()
        tokenizer = self._processor.tokenizer
        self._ids = (
            tokenizer.convert_tokens_to_ids("possible"),
            tokenizer.convert_tokens_to_ids("not"),
        )

    def _generate(self, image: SanitisedImage, prompt: str) -> tuple[str, list[float] | None]:
        """The raw answer and one score per status value (None if they cannot be paired)."""
        with Image.open(io.BytesIO(image.data)) as opened:
            picture = fit_image(opened.convert("RGB"), self._image_max_px)
        turn = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(turn, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[picture], return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            out = self._net.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )
        new_ids = out.sequences[0, inputs["input_ids"].shape[1] :]
        tokenizer = self._processor.tokenizer
        raw = str(tokenizer.decode(new_ids, skip_special_tokens=True))
        if not out.logits:
            return raw, None
        probs = torch.softmax(torch.stack([step[0] for step in out.logits]).float(), dim=-1)
        p_possible, p_not = (probs[:, i].tolist() for i in self._ids)
        tokens = [tokenizer.decode([i]) for i in new_ids.tolist()]
        try:
            return raw, status_scores(tokens, p_possible, p_not)
        except ValueError:
            return raw, None

    async def analyse(self, image: SanitisedImage, modality: Modality) -> VisionOutput:
        prompt = build_prompt(self._release.assessed & ASSESSABLE[modality], modality)
        started = time.perf_counter()
        raw, scores = await asyncio.to_thread(self._generate, image, prompt)
        elapsed_ms = (time.perf_counter() - started) * 1000
        paired = None if scores is None else scores_by_code(answer_codes(raw), scores)
        return parse_output(raw, paired, self._release, latency_ms=elapsed_ms)
