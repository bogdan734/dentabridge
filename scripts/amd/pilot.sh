#!/usr/bin/env bash
# The 50-step pilot on a fresh droplet: data, model, 50 steps, one run.json. Run after bootstrap.sh,
# from the repository root. AlphaDent downloads in the background for the real runs.
set -euo pipefail
source .commit.env 2>/dev/null || true
source .venv-amd/bin/activate
mkdir -p var/runs
python -m dentabridge_ml.data fetch aku_caries
python -m dentabridge_ml.data extract aku_caries
nohup python -m dentabridge_ml.data fetch alphadent > var/fetch-alphadent.log 2>&1 &
python -m dentabridge_ml.train sft train --sources aku_caries --out var/train-pilot.jsonl
python -m dentabridge_ml.train run packages/ml/configs/train/pilot-50.toml var/train-pilot.jsonl \
  --out var/runs/pilot-50 2>&1 | tee var/runs/pilot-50.log
cat var/runs/pilot-50/run.json
