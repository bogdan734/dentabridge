# ML1: can Ollama serve the tuned model?

Checked on 2026-10-25 with Ollama 0.40.10, llama.cpp commit `2280f0fd163b03d4331d49e08460f66502e35854`
(converter only, fetched by `scripts/llama-cpp.sh`), transformers 5.20.2, on an M3 Max.

## Route A: the safetensors directory (fails)

`ollama create` imports a safetensors directory as an MLX model. `--force` skips the validation
that rejects it ("unsupported MLX architecture: Qwen3VLForConditionalGeneration"), the model is
created, and the first request fails: `mlx runner failed: unsupported architecture`.
`--quantize q8_0` is refused (this build offers int4, int8, nvfp4, mxfp4, mxfp8 for safetensors).

## Route B: GGUF + projector (works)

`python -m dentabridge_ml.pilot gguf <checkpoint dir> --name <model> --out <dir>` runs llama.cpp's
`convert_hf_to_gguf.py` twice (language model as q8_0, 4.28 GB; vision projector as f16, 0.84 GB)
and registers both with a two-`FROM` Modelfile that names the instruct renderer and parser. The
stock `qwen3-vl:4b` of the Ollama library is the *thinking* variant, so its Modelfile is not reused.

Measured on the untuned Qwen3-VL-4B-Instruct converted this way:

| check | result |
|---|---|
| names the colour of a big red / green / blue square | red, green, blue (image reaches the model) |
| `logprobs` with `top_logprobs` | returned for every token, also under schema-constrained decoding |
| per-finding scores from those logprobs | found for every status value in the answers that parsed (for example 0.58, 0.33) |

## Not measured yet

Agreement with Transformers greedy decoding and the valid-JSON rate need a *tuned* model: the
untuned model rarely follows the output contract (41.7 % of 12 reference answers parsed), so its
answers say nothing about the import. They are measured on the first tuned adapter, with the
Transformers predictions of the training run as the reference:

```
python -m dentabridge_ml.train merge <adapter> --base data/models/Qwen3-VL-4B-Instruct --out var/tuned/merged
python -m dentabridge_ml.pilot gguf var/tuned/merged --name dentabridge-tuned --out var/tuned/gguf
python -m dentabridge_ml.pilot compare <reference val.predictions.jsonl> --name dentabridge-tuned --image-max-px 448 --out docs/results/ml/roundtrip.json
```

Provisional verdict: Ollama through route B. It stands if agreement is at least 0.90 and the
valid-JSON rate at least 0.95 on the tuned model; otherwise the tuned model is served by the
Transformers backend (CUDA, MPS, CPU) and Ollama keeps the baselines.
