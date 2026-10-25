#!/usr/bin/env bash
# Upload the code bundle as a private Kaggle dataset and push the notebook as a private kernel that
# runs the given stages on a T4 (default: the smoke test). Uses this machine's `kaggle` CLI login.
# Usage: scripts/kaggle/push.sh [stage...]     e.g. scripts/kaggle/push.sh setup data smoke train merge eval zeroshot
# Then: kaggle kernels status|logs|output <user>/dentabridge-lean
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
stages=("$@")
[ ${#stages[@]} -gt 0 ] || stages=(setup data smoke)
user="$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')"
[ -n "$user" ] || { echo "no kaggle login: run \`kaggle config view\`" >&2; exit 1; }

scripts/kaggle/bundle.sh
dataset=var/kaggle/dataset
kernel=var/kaggle/kernel
rm -rf "$dataset" "$kernel"
mkdir -p "$dataset" "$kernel"
cp var/kaggle/dentabridge-code.tar.gz "$dataset/"
printf '{"title": "dentabridge-code", "id": "%s/dentabridge-code", "licenses": [{"name": "other"}]}\n' "$user" \
  > "$dataset/dataset-metadata.json"
if kaggle datasets status "$user/dentabridge-code" >/dev/null 2>&1; then
  kaggle datasets version -p "$dataset" -m "$(git rev-parse --short HEAD)"
else
  kaggle datasets create -p "$dataset"
fi
until [ "$(kaggle datasets status "$user/dentabridge-code" 2>/dev/null | tail -1)" = "ready" ]; do sleep 5; done

python3 - "$user" "${stages[@]}" <<'PY'
import json, sys
user, stages = sys.argv[1], sys.argv[2:]
notebook = json.load(open("packages/ml/notebooks/kaggle_lean.ipynb", encoding="utf-8"))
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "".join(cell["source"]).lstrip().startswith("STAGES"):
        cell["source"] = [f"STAGES = {json.dumps(stages)}\n"]
json.dump(notebook, open("var/kaggle/kernel/kaggle_lean.ipynb", "w", encoding="utf-8"), indent=1)
json.dump(
    {
        "id": f"{user}/dentabridge-lean",
        "title": "dentabridge-lean",
        "code_file": "kaggle_lean.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [f"{user}/dentabridge-code"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    },
    open("var/kaggle/kernel/kernel-metadata.json", "w", encoding="utf-8"),
    indent=1,
)
PY
kaggle kernels push -p "$kernel" --accelerator NvidiaTeslaT4
echo "stages: ${stages[*]} | follow with: kaggle kernels status $user/dentabridge-lean"
