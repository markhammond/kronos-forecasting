#!/usr/bin/env bash
# Fetch the TimesFM 3.0 checkpoint (~1.2 GB).
#
# THE WEIGHTS ARE NOT OPEN SOURCE. They are covered by the TimesFM Non-Commercial
# License v1.0, which permits testing, evaluation and research, and forbids
# revenue-generating activity and production deployment. This repository's code is
# Apache-2.0; the checkpoint it loads is not. Read the licence before running this.
set -euo pipefail

REPO="google/timesfm-3.0-pytorch"
DIR="$(cd "$(dirname "$0")/.." && pwd)/checkpoints/timesfm-3.0-pytorch"
BASE="https://huggingface.co/$REPO/resolve/main"

echo "TimesFM 3.0 weights are NON-COMMERCIAL, research/evaluation only."
echo "Source: https://huggingface.co/$REPO"
echo

mkdir -p "$DIR"
for f in config.json LICENSE README.md model.safetensors; do
  if [ -s "$DIR/$f" ]; then echo "  have  $f"; continue; fi
  echo "  fetch $f"
  curl -fsSL --retry 3 -o "$DIR/$f" "$BASE/$f"
done

# A truncated 1.2 GB download loads without complaint until a tensor read runs off the
# end, so verify the declared extent against the file before trusting it.
python3 - "$DIR/model.safetensors" <<'PY'
import json, struct, sys, os
p = sys.argv[1]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode())
t = {k: v for k, v in hdr.items() if k != "__metadata__"}
end = 8 + n + max(v["data_offsets"][1] for v in t.values())
size = os.path.getsize(p)
print(f"  {len(t)} tensors; {'complete' if end == size else f'TRUNCATED ({end} != {size})'}")
sys.exit(0 if end == size else 1)
PY
