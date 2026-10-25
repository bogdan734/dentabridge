#!/usr/bin/env bash
# Build var/kaggle/dentabridge-code.tar.gz from the last commit, to upload as a private Kaggle
# dataset named "dentabridge-code" (the repository is private until it is published).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ -z "$(git status --porcelain)" ] || { echo "commit first: the bundle is built from HEAD" >&2; exit 1; }
out=var/kaggle
rm -rf "$out/tree"
mkdir -p "$out/tree"
git archive --prefix=dentabridge/ HEAD | tar -x -C "$out/tree"
git rev-parse --short HEAD > "$out/tree/dentabridge/COMMIT"
tar -czf "$out/dentabridge-code.tar.gz" -C "$out/tree" dentabridge
rm -rf "$out/tree"
echo "$out/dentabridge-code.tar.gz ($(du -h "$out/dentabridge-code.tar.gz" | cut -f1))"
