#!/usr/bin/env bash
# Copy the tracked files of this checkout to an AMD Developer Cloud droplet.
# Usage: scripts/amd/sync.sh root@<droplet-ip>
set -euo pipefail
target="${1:?usage: scripts/amd/sync.sh user@host}"
cd "$(git rev-parse --show-toplevel)"
commit="$(git rev-parse --short HEAD)"
[ -z "$(git status --porcelain)" ] || commit="$commit-dirty"
git ls-files -z | rsync -az --from0 --files-from=- ./ "$target:dentabridge/"
ssh "$target" "echo 'export DENTABRIDGE_COMMIT=$commit' > dentabridge/.commit.env"
echo "synced $commit to $target:dentabridge"
