#!/usr/bin/env bash
# Fetch the reference implementation the parity harness compares against.
#
# Not vendored: these are Google's files, Apache-2.0, and carrying copies would oblige
# this repository to track their notices and their upstream drift. Pinned to a revision
# because parity is only meaningful against a known implementation.
set -euo pipefail

REV="${TIMESFM_REV:-331c6d33cb1ac2611de3056d0ac7164aab6301eb}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/reference/timesfm3"
BASE="https://raw.githubusercontent.com/google-research/timesfm/$REV/src/timesfm3"

mkdir -p "$DIR"
: > "$DIR/__init__.py"
for f in model configs dense transformer util normalization cpm_revin_refine; do
  echo "  fetch $f.py"
  curl -fsSL --retry 3 -o "$DIR/$f.py" "$BASE/$f.py"
done

echo
echo "Fetched at revision $REV (Apache-2.0, Copyright Google LLC)."
echo "Generate parity tensors with:"
echo "  .venv/bin/python reference/dump_parity.py"
