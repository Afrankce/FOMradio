#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/ka/CKM/radio-unet
PYTHON_BIN=/home/ka/anaconda3/envs/radio-unet-w32/bin/python
DATASET_ROOT=/home/ka/CKM/data/raw/49ca1dcebe2caa2b2112e6c862132243a992b00a/Dataset
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/home/ka/CKM/radio-unet/runs/width32_b128_20260908T052359Z}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
MANIFEST_DIR=${MANIFEST_DIR:-$EXPERIMENT_ROOT/manifests}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
WORLD_SIZE=${WORLD_SIZE:-2}
cd "$REPO_ROOT"

for array in 8x8 16x16 32x32; do
  mkdir -p "$EXPERIMENT_ROOT/$array"
  train_command=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" \
    "$REPO_ROOT/run_width32_distributed.py" \
    --dataset-root "$DATASET_ROOT" \
    --manifest-path "$MANIFEST_DIR/manifest_samefreq_6.7ghz_${array}_0deg.jsonl" \
    --height-stats-path "$MANIFEST_DIR/height_stats_train.json" \
    --run-root "$EXPERIMENT_ROOT/$array" \
    --array-size "$array" --resume auto \
    --train-batch-size "$TRAIN_BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE")
  train_needed=1
  if [[ -f "$EXPERIMENT_ROOT/$array/training_runtime.json" ]] && \
     "$PYTHON_BIN" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "complete")' "$EXPERIMENT_ROOT/$array/training_runtime.json"; then
    train_needed=0
  fi
  if { [[ "$train_needed" == 0 ]] || "${train_command[@]}" >> "$EXPERIMENT_ROOT/$array/launcher.log" 2>&1; } && \
     "${train_command[@]}" --test-only >> "$EXPERIMENT_ROOT/$array/launcher.log" 2>&1; then
    continue
  fi
  if [[ "$TRAIN_BATCH_SIZE" == 128 ]] && rg -q 'CUDA out of memory|torch.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY' "$EXPERIMENT_ROOT/$array/launcher.log"; then
    echo "OOM in $array: restarting all arrays from scratch with train=64/eval=128; old artifacts preserved."
    EXPERIMENT_ROOT="${EXPERIMENT_ROOT}_fallback64" MANIFEST_DIR="$MANIFEST_DIR" \
      TRAIN_BATCH_SIZE=64 EVAL_BATCH_SIZE=128 bash "$REPO_ROOT/scripts/run_width32_arrays.sh"
    exit $?
  fi
  echo "Training failed for $array; inspect $EXPERIMENT_ROOT/$array/launcher.log" >&2
  exit 1
done
