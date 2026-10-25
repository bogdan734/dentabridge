#!/usr/bin/env bash
# Copy run results (adapters, run.json, logs, predictions; not checkpoints) back from the droplet.
# Usage: scripts/amd/fetch.sh root@<droplet-ip>
set -euo pipefail
target="${1:?usage: scripts/amd/fetch.sh user@host}"
key="${AMD_SSH_KEY:-$HOME/.ssh/id_ed25519_dentabridge}"
ssh_cmd="ssh -i $key -o IdentitiesOnly=yes"
cd "$(git rev-parse --show-toplevel)"
mkdir -p var/amd
rsync -az -e "$ssh_cmd" --exclude 'checkpoint-*' "$target:dentabridge/var/runs/" var/amd/runs/
rsync -az -e "$ssh_cmd" "$target:dentabridge/var/eval/" var/amd/eval/ || true
echo "fetched into var/amd"
