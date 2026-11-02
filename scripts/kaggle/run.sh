#!/usr/bin/env bash
# The stages of a T4 training run on Kaggle (or Colab). The notebook only calls these.
#
#   scripts/kaggle/run.sh <stage>
#
#   setup     uv, Python 3.13, CUDA torch, the project, bitsandbytes
#   data      download, unpack and verify the datasets; fetch the base weights once for all lanes
#   smoke     ten steps of the lean recipe: seconds per step and the full-run estimate, and a
#             broken install shows up here instead of after an hour
#   train     CONFIG on the training rows; resumes from the newest checkpoint of RUN if there is one
#   merge     fold the adapter into the base weights (fp16: a T4 has no bf16)
#   eval      score the merged model on the validation split, then drop the merged weights
#   zeroshot  score the untuned Instruct model with the same prompt (the baseline)
#
# Settings (environment): CONFIG (default lean-t4), RUN (default = CONFIG), GPU (default 0), REPEAT
# (default alphadent=3: AlphaDent is 16 % of the training photos and the only source of filling,
# crown and abrasion, so its photos go in three times; "" turns it off).
# A two-GPU session runs one lane per GPU, each with its own CONFIG, RUN and GPU: see the notebook.
# Kaggle paths are the defaults; on Colab set WORK=/content SCRATCH=/content/scratch ROOT=/content/dentabridge.
set -euo pipefail

WORK="${WORK:-/kaggle/working}"        # kept as the notebook's output: adapters, results
SCRATCH="${SCRATCH:-/kaggle/temp}"     # thrown away with the session: venv, datasets, merged weights
ROOT="${ROOT:-$WORK/dentabridge}"
CONFIG="${CONFIG:-lean-t4}"
RUN="${RUN:-$CONFIG}"
REPEAT="${REPEAT-alphadent=3}"
OUT="$WORK/out"
VENV="$SCRATCH/venv"
PY="$VENV/bin/python"
BASE="Qwen/Qwen3-VL-4B-Instruct"
export DENTABRIDGE_DATA_ROOT="$SCRATCH/data"
export HF_HOME="$SCRATCH/hf"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$SCRATCH"
cd "$ROOT"
[ -f COMMIT ] && export DENTABRIDGE_COMMIT="$(cat COMMIT)"
SOURCES=(aku_caries alphadent)

image_px() { "$PY" -c "import sys,tomllib; print(tomllib.load(open(sys.argv[1],'rb'))['image_max_px'])" "packages/ml/configs/train/$CONFIG.toml"; }

# --no-config: the repository pins the uv version contributors use; Kaggle's own uv is newer.
uvc() { uv --no-config "$@"; }

stage_setup() {
  command -v uv >/dev/null || pip install -q uv
  command -v bsdtar >/dev/null || { apt-get update -qq && apt-get install -y -qq libarchive-tools; }
  uvc venv --quiet --python 3.13 "$VENV"
  uvc pip install --quiet --python "$PY" torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126
  uvc pip install --quiet --python "$PY" -e packages/core -e packages/ml \
    transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 bitsandbytes==0.50.2
  "$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "no GPU: choose a GPU accelerator in the notebook settings"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
PY
}

build_rows() {
  # Lanes start together: write beside, then rename, so nobody reads half a file. Same seed, same rows.
  [ -f "$OUT/train.jsonl" ] && return
  repeat=()
  for pair in $REPEAT; do repeat+=(--repeat "$pair"); done
  "$PY" -m dentabridge_ml.train sft train "${repeat[@]}" --out "$OUT/train.jsonl.$$"
  mv "$OUT/train.jsonl.$$" "$OUT/train.jsonl"
}

stage_data() {
  "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$BASE')"
  if "$PY" -m dentabridge_ml.data verify >/dev/null 2>&1; then echo "data already in place"; return; fi
  "$PY" -m dentabridge_ml.data fetch "${SOURCES[@]}"
  "$PY" -m dentabridge_ml.data extract "${SOURCES[@]}"
  "$PY" -m dentabridge_ml.data verify
  rm -rf "$DENTABRIDGE_DATA_ROOT/raw"  # the archives: the session disk is finite
}

stage_smoke() {
  build_rows
  head -n 160 "$OUT/train.jsonl" > "$OUT/smoke.jsonl"
  "$PY" -m dentabridge_ml.train run packages/ml/configs/train/smoke-t4.toml "$OUT/smoke.jsonl" --out "$OUT/smoke"
  "$PY" - "$OUT" "$CONFIG" <<'PY'
import json, sys, tomllib
out, config = sys.argv[1], sys.argv[2]
run = json.load(open(f"{out}/smoke/run.json"))
rows = sum(1 for _ in open(f"{out}/train.jsonl"))
epochs = tomllib.load(open(f"packages/ml/configs/train/{config}.toml", "rb"))["epochs"]
hours = rows * epochs / run["samples_per_second"] / 3600
print(f"{run['samples_per_second']:.2f} samples/s -> the {rows}-row, {epochs}-epoch run takes about {hours:.1f} h")
PY
}

stage_train() {
  build_rows
  resume=()
  compgen -G "$OUT/$RUN/checkpoint-*" >/dev/null && resume=(--resume)
  "$PY" -m dentabridge_ml.train run "packages/ml/configs/train/$CONFIG.toml" "$OUT/train.jsonl" \
    --out "$OUT/$RUN" "${resume[@]}"
  rm -rf "$OUT/$RUN"/checkpoint-*  # only for resuming; the adapter is saved
}

stage_merge() {
  "$PY" -m dentabridge_ml.train merge "$OUT/$RUN/adapter" --base "$BASE" \
    --out "$SCRATCH/merged-$RUN" --dtype fp16
}

stage_eval() {
  "$PY" -m dentabridge_ml.eval run --backend "transformers:$SCRATCH/merged-$RUN" --split val \
    --image-max-px "$(image_px)" --batch-size 8 --out-dir "$OUT/$RUN/val"
  rm -rf "$SCRATCH/merged-$RUN"  # 8 GB; the Mac merges the adapter again from the base weights
}

stage_zeroshot() {
  "$PY" -m dentabridge_ml.eval run --backend "transformers:$BASE" --split val \
    --image-max-px "$(image_px)" --batch-size 8 --out-dir "$OUT/zeroshot/val"
}

stage="${1:?usage: scripts/kaggle/run.sh setup|data|smoke|train|merge|eval|zeroshot}"
case "$stage" in
  setup|data|smoke|train|merge|eval|zeroshot) "stage_$stage" ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
