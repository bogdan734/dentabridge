#!/usr/bin/env bash
# Copy the tracked files of this checkout to an AMD Developer Cloud droplet.
# Usage: scripts/amd/sync.sh root@<droplet-ip>    (key: $AMD_SSH_KEY, default ~/.ssh/id_ed25519_dentabridge)
set -euo pipefail
target="${1:?usage: scripts/amd/sync.sh user@host}"
key="${AMD_SSH_KEY:-$HOME/.ssh/id_ed25519_dentabridge}"
ssh_cmd="ssh -i $key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
cd "$(git rev-parse --show-toplevel)"
commit="$(git rev-parse --short HEAD)"
[ -z "$(git status --porcelain)" ] || commit="$commit-dirty"
git ls-files -z | rsync -az --from0 --files-from=- -e "$ssh_cmd" ./ "$target:dentabridge/"
$ssh_cmd "$target" "echo 'export DENTABRIDGE_COMMIT=$commit' > dentabridge/.commit.env"
echo "synced $commit to $target:dentabridge"
