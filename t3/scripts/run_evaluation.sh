#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-2}"
SPLIT="${1:-test}"
cd "$ROOT/code"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m beam_boundary.evaluate --root "$ROOT" --arm T3 --base E0 \
  --split "$SPLIT" --checkpoint-dir "$ROOT/jobs/T3"

