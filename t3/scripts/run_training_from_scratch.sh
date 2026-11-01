#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-2}"
OUT="${OUT:-$ROOT/experiments/retrain_seed42}"
STOP="${STOP:-5000}"
cd "$ROOT/code"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m beam_boundary.train --root "$ROOT" --arm T3 --base E0 \
  --stop "$STOP" --out "$OUT"

