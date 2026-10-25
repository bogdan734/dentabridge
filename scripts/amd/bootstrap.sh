#!/usr/bin/env bash
# Run on the droplet, from the repository root. Builds a Python 3.13 environment whose PyTorch is
# the ROCm 6.4 wheel, matching the Quick Start image's host driver (amdgpu-dkms 6.12, ROCm 6.4).
set -euo pipefail
python3 -m venv ~/uv-tool
~/uv-tool/bin/pip install --quiet "uv>=0.11.19,<0.12"
UV=~/uv-tool/bin/uv
$UV venv --python 3.13 .venv-amd
$UV pip install --python .venv-amd/bin/python "torch==2.9.1" "torchvision==0.24.1" \
  --index-url https://download.pytorch.org/whl/rocm6.4
# the pins are the versions the code was tested with; torch stays the ROCm build above
$UV pip install --python .venv-amd/bin/python -e packages/core -e packages/ml \
  "transformers==5.20.2" "peft==0.21.2" "accelerate==1.15.0"
.venv-amd/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible: check the image and the driver"
print("device", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30), "GiB")
PY
