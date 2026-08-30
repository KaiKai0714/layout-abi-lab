#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_TAG="${LAYOUTABI_DEVICE_TAG:-$(hostname)}"
OUTPUT="${LAYOUTABI_OUTPUT:-$ROOT/results/local_${DEVICE_TAG}}"

cd "$ROOT"
python -m layoutabi.cli reproduce --output "$OUTPUT" "$@"
python -m layoutabi.cli validate "$OUTPUT"

echo "Validated result bundle: $OUTPUT"

