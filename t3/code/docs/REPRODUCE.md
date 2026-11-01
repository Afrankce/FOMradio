# Reproducing the fixed-protocol experiment

## Fixed protocol

- Frequency: 6.7 GHz.
- Steering: one 0-degree beam.
- Arrays: 8x8, 16x16, and 32x32.
- Scene split: seed 42, train/validation/test = 560/80/160, with disjoint scenes.
- Working resolution: 256x256.
- Conditions: Tx mask, height, and beam map.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-5`.
- At most 1,000 epochs, warmup ratio 0.1, early-stopping patience 20.
- Micro-batch 2, accumulation 28, effective batch 56.
- EMA decay 0.999 and FP16 AMP.
- CFG condition dropout 0.25 during training and CFG=1.0 at evaluation.
- Fixed hash-based initial noise and two-step Euler integration.

An immutable configuration class validates all critical values. Do not modify locked constants simply to make a run start: that would invalidate the identity hash of existing checkpoints.

## Directory layout

```text
/data/multiconfig/                  # Dataset root
/data/manifests/
  scene_split_seed42.json
  height_stats_train.json
  manifest_samefreq_6.7ghz_8x8_0deg.jsonl
  manifest_samefreq_6.7ghz_16x16_0deg.jsonl
  manifest_samefreq_6.7ghz_32x32_0deg.jsonl
/runs/uno/{8x8,16x16,32x32}/        # config.json, best.pt, last.pt, metrics.csv
/results/uno/{8x8,16x16,32x32}/     # Final metrics, predictions, figures
```

Directory names may change, but relative paths within manifests must still resolve under the dataset root.

## 1. Create the environment

From the FOMradio code checkout (the historical training branch was `release/multiscale-uno-fm`):

```bash
cd FOMradio/code
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch for the local driver/CUDA combination first.
python -m pip install -r requirements.txt
```

The reference training environment used PyTorch 2.5.1 with CUDA 12.1. Other machines need not use an identical driver, but a paper-level reproduction should record Python, PyTorch, CUDA, GPU model, and the current Git commit.

## 2. Build three single-beam manifests

Place `scene_split_seed42.json` in the manifest directory and run the builder for each array:

```bash
python prepare_same_frequency.py build-manifest \
  --dataset-root /data/multiconfig \
  --split-path /data/manifests/scene_split_seed42.json \
  --array-size 8x8 \
  --frequency-hz 6700000000 \
  --steering-deg 0 \
  --output /data/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl
```

Replace both occurrences of `8x8` with `16x16` and `32x32` in subsequent commands. The builder uses the frozen schema to select the actual 0-degree beam ID for each array and validates all 800 samples and the fixed split.

`height_stats_train.json` must be computed only from the same 560 training scenes and contain hash evidence for each height file. Training preflight validates its schema, scene count, and split SHA-256.

## 3. Run preflight

```bash
python run_same_frequency_multiscale_uno.py train \
  --dataset-root /data/multiconfig \
  --manifest-path /data/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl \
  --height-stats-path /data/manifests/height_stats_train.json \
  --run-root /runs/uno/8x8 \
  --array-size 8x8 \
  --device cuda:0 \
  --resume none \
  --preflight-only
```

Preflight loads the first sample from each split and checks the `[3,256,256]` condition, `[1,256,256]` target, nonempty valid mask, data version, manifest hash, Git ancestry, and origin.

## 4. Train or resume

For a new run:

```bash
python run_same_frequency_multiscale_uno.py train \
  --dataset-root /data/multiconfig \
  --manifest-path /data/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl \
  --height-stats-path /data/manifests/height_stats_train.json \
  --run-root /runs/uno/8x8 \
  --array-size 8x8 \
  --device cuda:0 \
  --resume none
```

For an interrupted run, change the final option to `--resume auto`.

`last.pt` stores the model, EMA, optimizer, scheduler, AMP scaler, trainer state, and training-data generator state. Resume performs a strict identity check so that a checkpoint from another array or configuration cannot be attached accidentally.

The arrays may train concurrently, one GPU each:

```bash
DATASET_ROOT=/data/multiconfig \
MANIFEST_DIR=/data/manifests \
HEIGHT_STATS=/data/manifests/height_stats_train.json \
RUN_ROOT=/runs/uno \
RESULTS_ROOT=/results/uno \
bash scripts/run_three_arrays.sh train
```

Set `GPU_8X8`, `GPU_16X16`, or `GPU_32X32` to override the default GPU indices `0/1/2`.

## 5. Freeze validation choice and evaluate test

Run `select-cfg` for every array before `test`. It checks the locked candidate `CFG=1.0` on all 80 validation scenes and binds the choice to the epoch and identity of `best.pt`. Then evaluate the 160 test scenes once:

```bash
bash scripts/run_three_arrays.sh evaluate
```

The final directory contains:

- `metrics_overall.json` and a per-frequency summary CSV;
- one prediction artifact per test sample;
- comparisons and error figures for fixed scenes;
- a runtime benchmark;
- `run_manifest.json` and CFG/checkpoint hash evidence.

Formal result directories refuse to overwrite existing outputs. Use a new output directory for any rerun and preserve the old one as an immutable experiment record.

## 6. Metric definitions

- dB-RMSE and dB-MAE: propagation pixels where `valid_mask` is true.
- NMSE: dataset-total squared error divided by dataset-total target energy.
- PSNR: computed in the normalized `[0,1]` domain.
- SSIM: standard local-window implementation, accumulated under the valid-region convention.
- All test metrics use EMA `best.pt`, fixed noise, CFG=1.0, and two-step Euler integration.
