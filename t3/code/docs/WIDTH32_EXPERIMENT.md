# Width=32 ablation — launch specification

The UNO operator width changes from 24 to 32. At the user's subsequent request,
global training batch also changes from 56 to 128 and evaluation batch to 256.
This is therefore no longer a strictly width-only comparison with the old run.
Existing width-24 model,
configuration, factory locks and checkpoints remain unchanged. The actual release
fork URL is now accepted by provenance validation, still requiring ancestry from
the pinned RadioFlow upstream commit. The actual origin and code hashes are recorded.

## Data and frozen scientific parameters

- Raw root: `/home/ka/CKM/data/raw/49ca1dcebe2caa2b2112e6c862132243a992b00a/Dataset` (read-only).
- Arrays: separate 8x8, 16x16, 32x32 runs, executed sequentially on the selected GPU group.
- Frequency: 6.7 GHz; steering: 0 degrees; actual beam IDs resolved from the frozen schema.
- Scene split: original `python_random_v1`, seed 42, 560/80/160 train/validation/test.
- Height normalization: maximum from the 560 training scenes only.
- Input/output grid: 256x256; original target preprocessing and valid-pixel masks unchanged.
- State channels: [32,64,128,256,256]; operator width: 32 at all nine stages.
- Modes: [12,12,8,4,4]; padding: [9,5,3,2,1].
- CFM sigma=0; valid-pixel-weighted velocity MSE; condition dropout=0.25.
- AdamW: peak LR=0.001, weight decay=0.00001.
- Training batch 128, validation/test batch 256; OOM fallback: 64/128.
- Warmup: 500 successful optimizer updates, then cosine schedule, 5000 planned updates.
- At fallback batch 64: warmup 900 updates, 9000 planned updates. Warmup remains 10%.
- Maximum 1000 epochs; patience=20; EMA decay=0.999.
- AMP FP16; FFT/complex contractions FP32/complex64.
- Validation: 80 scenes every epoch; EMA, CFG=1, fixed sample noise, 2-step Euler.
- Test set is not scored during training or used to adjust hyperparameters.
- After each completed run, the queue evaluates the best EMA once on all 160 test
  scenes with batch limit 256 (128 on fallback), recording `test_metrics.json`.
  A rerun reuses existing test metrics only if config and checkpoint hashes match.

## Environment and parallel execution

Conda environment: `radio-unet-w32`, Python 3.11, PyTorch 2.5.1+cu124,
torchvision 0.20.1+cu124, MONAI 1.4.0, torchcfm 1.0.7, NumPy 1.26.4.
The resolved full environment is saved under the experiment root before launch.

Initial server check: 8 A100 80GB cards; only physical GPUs 5 and 6 were completely
idle. User explicitly approved launching on those two cards. Other jobs are not
stopped or placed in contention deliberately.

Two ranks train ONE model synchronously: local micro-batch 64 x 2 GPUs, no
accumulation = global batch 128, 5 optimizer windows per epoch (last batch 48).
Four GPUs are supported with local batch 32. Do not change rank count mid-run.
Fallback uses local batch 32 x 2 GPUs = global 64, 9 updates/epoch (last batch 48).
Validation has only 80 samples and test has 160; batch 256 is an upper limit,
not a reason to repeat samples. A separate 256-sample validation-input repeat
checks worst-case inference memory without evaluating test labels.
Each epoch uses all 560 train scenes exactly once, with no sampler padding.
Per-rank RNG streams and epoch-indexed distributed shuffling differ from the old
single-GPU execution; this is not a claim of bitwise reproduction of that run.

This uses an explicit NCCL gradient SUM on real views of complex tensors, not
the stock DDP wrapper. Losses normalize by the global valid-pixel count, not by
rank-local counts. AMP checks occur after synchronization, so overflow handling
is shared by all ranks. Rank 0 evaluates validation and broadcasts its metrics.

## Commands (working directory `/home/ka/CKM/radio-unet`)

```bash
conda activate radio-unet-w32
CUDA_VISIBLE_DEVICES="" python /home/ka/CKM/radio-unet/prepare_width32.py \
  --dataset-root /home/ka/CKM/data/raw/49ca1dcebe2caa2b2112e6c862132243a992b00a/Dataset \
  --output /home/ka/CKM/radio-unet/runs/width32_b128_20260908T052359Z/manifests

CUDA_VISIBLE_DEVICES=5,6 WORLD_SIZE=2 TRAIN_BATCH_SIZE=128 EVAL_BATCH_SIZE=256 \
  bash /home/ka/CKM/radio-unet/scripts/run_width32_arrays.sh
```

The script expands to one `torch.distributed.run --standalone --nproc_per_node=2`
command per array. Each command uses `run_width32_distributed.py --array-size ARRAY
--dataset-root RAW_ROOT --manifest-path MANIFEST --height-stats-path STATS
--run-root ARRAY_RUN_ROOT --resume auto --train-batch-size 128 --eval-batch-size 256`.

Before formal launch: CPU numerical gradient/parameter/sampler tests, source
preflight, then `--smoke-optimizer-steps 2 --capacity-check` on the two selected GPUs. Smoke artifacts
are isolated under `_smoke`; formal runs initialize from scratch, not smoke weights.

## Outputs and recovery

Experiment root: `/home/ka/CKM/radio-unet/runs/width32_b128_20260908T052359Z`.
Earlier batch-56 setup/smoke/checkpoint trials remain in
`/home/ka/CKM/radio-unet/runs/width32_20260908T051759Z`; they are not final runs.
Each array has `config.json`, `metrics.csv`, `best.pt`, `last.pt`, `launcher.log`,
`committed_epoch.json`, and completion/paused `training_runtime.json`.
Nonzero ranks keep their own full RNG/optimizer checkpoints in `rank_N/`.
All ranks must be retained for exact same-topology resume. An incomplete rank
checkpoint transaction is rejected, not silently resumed. The source digest is
part of the scientific identity; changes to training code require a new run.

On an explicit CUDA OOM, the queue restarts all three arrays from scratch in
an independent `_fallback64` root with batches 64/128, preserving the old results.
Other failures stop the queue; a second OOM at 64/128 also stops it.
Re-running the same launch command
resumes existing runs; do not overwrite or delete rank checkpoints manually.
Training start is not experimental completion. Comparison with width 24 remains
pending until the new runs finish and their evaluation protocol is finalized.
