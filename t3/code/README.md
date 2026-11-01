# FOMradio: Multiscale UNO-FM code

This research code uses an **attention-conditioned multiscale UNO** velocity network for conditional flow matching (CFM) on 6.7 GHz, 0-degree, single-beam radio-map prediction. The documented release covers `8x8`, `16x16`, and `32x32` arrays under one fixed 560/80/160 scene-level train/validation/test split.

> This is an experimental extension, not a line-by-line reproduction of the original RadioFlow paper. It retains the RadioFlow-style condition encoder and CA/SA fusion while replacing the state-velocity backbone with a U-shaped multiscale FNO (UNO).

![FOMradio Multiscale UNO-FM architecture](assets/architecture.png)

## Scope

- Condition: `c = [Tx mask, height, beam map]`, shape `[B,3,256,256]`.
- Flow-matching state: `x_t`, shape `[B,1,256,256]`.
- Network output: velocity field `v_theta(x_t,t,c)`, shape `[B,1,256,256]`; it is not the final radio map.
- Condition encoder: five-scale `BasicUNetEncoder`.
- State backbone: four encoder FNO stages, one bottleneck stage, and four decoder FNO stages.
- Inference: EMA weights, fixed hash-based noise, CFG=1.0, and two-step Euler integration.
- Training: AdamW, FP16 AMP, gradient accumulation, early stopping, and strict checkpoint-identity checks.

The repository does not include the dataset, checkpoints, training outputs, caches, or a test directory. `.gitignore` excludes these files.

## Environment

Python 3.10+ and a CUDA-capable PyTorch installation are recommended. Install PyTorch for the local CUDA version first, followed by the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-compatible torch build from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

## Data preparation

`experiments/provenance.py` records the locked data source and version; `experiments/multiconfig_schema.json` specifies the structural data contract. In addition to the dataset root, prepare:

- `scene_split_seed42.json`: fixed 560/80/160 scene split;
- `height_stats_train.json`: height-normalization statistics computed only from the 560 training scenes;
- one 800-record JSONL manifest per array for 6.7 GHz and 0 degrees.

For example, build the 8x8 manifest:

```bash
python prepare_same_frequency.py build-manifest \
  --dataset-root /path/to/MultiConfigRadiomap \
  --split-path /path/to/manifests/scene_split_seed42.json \
  --array-size 8x8 \
  --frequency-hz 6700000000 \
  --steering-deg 0 \
  --output /path/to/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl
```

Repeat for `16x16` and `32x32`. The 0-degree beam IDs differ by array. The code resolves and checks them against the frozen schema; do not assume one shared beam ID.

## Training and evaluation

First run the read-only data preflight:

```bash
python run_same_frequency_multiscale_uno.py train \
  --dataset-root /path/to/MultiConfigRadiomap \
  --manifest-path /path/to/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl \
  --height-stats-path /path/to/manifests/height_stats_train.json \
  --run-root /path/to/runs/8x8 \
  --array-size 8x8 \
  --device cuda:0 \
  --resume none \
  --preflight-only
```

Remove `--preflight-only` to start training. After an interruption, use `--resume auto` for strict recovery from `last.pt`. The fixed configuration permits up to 1,000 epochs, early-stopping patience 20, micro-batch 2, and 28 accumulation steps (effective batch 56).

After training, freeze the validation-set CFG selection before running the test evaluation once:

```bash
python run_same_frequency_multiscale_uno.py select-cfg \
  --dataset-root /path/to/MultiConfigRadiomap \
  --manifest-path /path/to/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl \
  --height-stats-path /path/to/manifests/height_stats_train.json \
  --run-root /path/to/runs/8x8 \
  --results-root /path/to/results/8x8 \
  --array-size 8x8 \
  --device cuda:0

python run_same_frequency_multiscale_uno.py test \
  --dataset-root /path/to/MultiConfigRadiomap \
  --manifest-path /path/to/manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl \
  --height-stats-path /path/to/manifests/height_stats_train.json \
  --run-root /path/to/runs/8x8 \
  --results-root /path/to/results/8x8 \
  --array-size 8x8 \
  --device cuda:0
```

`results-root` must not exist before the formal test run. The evaluator writes metrics, per-sample predictions, and visualizations as an atomic directory transaction. See `scripts/run_three_arrays.sh` for a three-GPU parallel example.

## Documentation

- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Reproduction instructions](docs/REPRODUCE.md)
- [Completed results](docs/RESULTS.md)
- [Release cleanup](RELEASE_NOTES.md)

## Code entry points

```text
run_same_frequency_multiscale_uno.py   # Unified train / select-cfg / test entry point
prepare_same_frequency.py              # Build the locked manifest
model/attention_multiscale_uno.py      # UNO-FM model
model/fno.py                           # 2-D spectral convolution
training/                              # Configuration, optimization, EMA, checkpoints, training loop
evaluation/                            # Two-step Euler, metrics, visualizations
data_loaders/                          # Data contract, normalization, valid mask
```

## Result summary

The following values come from one server run under the fixed protocol; see [docs/RESULTS.md](docs/RESULTS.md) for interpretation limits.

| Array | dB-RMSE ↓ | dB-MAE ↓ | NMSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|
| 8x8 | 9.2108 | 5.5413 | 0.0026568 | 30.2565 | 0.9099 |
| 16x16 | 9.4604 | 5.6524 | 0.0029507 | 30.0243 | 0.9043 |
| 32x32 | 9.6818 | 5.8408 | 0.0034835 | 29.8233 | 0.8937 |

## License

See the repository-root [LICENSE](LICENSE).
