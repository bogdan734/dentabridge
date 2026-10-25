#!/usr/bin/env bash
# Fetch the parts of llama.cpp that convert a Hugging Face checkpoint to GGUF (about 3 MB, sparse
# clone) into var/tools/llama.cpp, pinned to the commit the Ollama round-trip was verified with.
# Usage: scripts/llama-cpp.sh [destination]
set -euo pipefail
sha=2280f0fd163b03d4331d49e08460f66502e35854
dest="${1:-var/tools/llama.cpp}"
if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD)" = "$sha" ]; then
  echo "llama.cpp $sha already in $dest"; exit 0
fi
rm -rf "$dest"
git clone --quiet --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp "$dest"
git -C "$dest" sparse-checkout set --no-cone /convert_hf_to_gguf.py /conversion /gguf-py
git -C "$dest" checkout --quiet "$sha"
echo "llama.cpp $sha in $dest"
