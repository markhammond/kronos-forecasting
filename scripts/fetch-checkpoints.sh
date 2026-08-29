#!/usr/bin/env bash
# Fetch the pinned checkpoints the weights packages embed.
#
# They are not committed: the Small pair alone is ~110 MB and git history is append-only.
# Revisions are pinned rather than tracking a branch, because published checkpoints get
# refreshed and a silent weight change would alter every downstream result while leaving
# the consuming code identical.
set -euo pipefail

ROOT="${KRONOS_CHECKPOINT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints}"
BASE="https://huggingface.co"

fetch () {
  local repo=$1 rev=$2
  local dir="$ROOT/models--${repo//\//--}/snapshots/$rev"
  mkdir -p "$dir"
  for file in config.json model.safetensors; do
    if [ -s "$dir/$file" ]; then
      echo "  have    $repo/$file"
      continue
    fi
    echo "  fetch   $repo/$file"
    curl -fsSL --retry 3 -o "$dir/$file" "$BASE/$repo/resolve/$rev/$file"
  done
}

echo "checkpoint root: $ROOT"
fetch NeoQuasar/Kronos-small            901c26c1332695a2a8f243eb2f37243a37bea320
fetch NeoQuasar/Kronos-Tokenizer-base   0e0117387f39004a9016484a186a908917e22426
fetch NeoQuasar/Kronos-mini             f4e68697d9d5aed55cef5c96aabc3376bcad9f81
fetch NeoQuasar/Kronos-Tokenizer-2k     26966d0035065a0cae0ebad7af8ece35bc1fb51c

echo
echo "Build the weights packages with:"
echo "  dotnet build -p:KronosCheckpointRoot=$ROOT"
