> This public source snapshot omits the dataset, prepared caches, and the T3 checkpoint. References below to included weights describe the original local package, not this snapshot.

# FOMradio

FOMradio is a research-code package for source-observation-conditioned flow matching on U6G multi-configuration radiomaps. This is an English-language, reorganized copy of the archived **T3 cross-array experiment**. Its checkpoint predicts a 32 × 32 target-array power map from a complete 8 × 8 source-array observation at 6.7 GHz. Renaming the package does **not** rename the checkpoint, change its training protocol, or make it a T2 checkpoint.

This README is aligned with the reproducibility and anonymity guidance in the [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines). ICLR does not prescribe a repository README template. This local package is **not yet cleared for double-blind upload**; see [Submission audit](#submission-audit).

## Method and scope

The six input channels, in order, are `Tx`, `Height`, `Beam32_target`, `Beam8_0deg`, `Y8_0deg`, and `Y8_valid`. The target is the normalized 32 × 32-array power map `Y32` on a 256 × 256 grid. A complete, measured 8 × 8 source map at 0° is assumed available for each scene. The source map is not an inferred or sparse observation.

Let `mu8 = where(Y8_valid, Y8_0deg, 0)` and let `z` be a multi-scale Matérn Gaussian-random-field draw. The archived T3 flow uses

```text
x0 = mu8 + z
xt = (1 - t) * x0 + t * Y32
ut = Y32 - mu8 - z
```

The velocity backbone is a six-channel `SourceObservationUNO` with no training-time classifier-free condition dropout (`cfg_drop_prob=0`). The loss is masked flow-matching MSE plus `0.05 * Haar1 + 0.05 * Haar2` after 2 × average pooling. The model does not use EMA. The archived run used seed 42, 560/80/160 scene splits, training target angles −14°/0°/+14°, two GPUs, effective global batch 128, and 5,000 successful optimizer updates. Evaluation uses Euler NFE 8 and four stochastic draws per case. See [`config.json`](config.json) and [`provenance/ORIGINAL_PROTOCOL.md`](provenance/ORIGINAL_PROTOCOL.md) for the frozen scientific contract.

This is **not** the later experiment in which training targets are only 8 × 8/0° and all 32 × 32 angles are zero-shot. It must not be used as that experiment's checkpoint or compared under an identical-training-target claim.

## Repository map

| Path | Purpose |
| --- | --- |
| `code/` | Model, data, training, and evaluation modules. Historical ablation modules remain for provenance. |
| `scripts/` | Data preparation, inference, training, evaluation, and integrity commands. |
| `configs/source.json` | Multi-scale Matérn source specification. |
| `manifests/` | Frozen train, validation, and test record lists. |
| `checkpoints/t3_step5000.pt` | Archived step-5,000 T3 checkpoint. |
| `jobs/T3/` | Compatibility location for the archived evaluation entry point. |
| `provenance/` | Original protocol and training history; not a claim that edited source bytes equal the original snapshot. |
| `data/` | Prepared-cache location; raw U6G data are not redistributed. |
| `tests/` | Contract tests. |

## Environment

Python 3.11 and a compatible CUDA-enabled PyTorch build are recommended. The archived environment used PyTorch 2.5.1+cu124, MONAI 1.4.0, and NumPy 1.26.4. Install a PyTorch build suitable for the machine first, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, replace the activation command as appropriate. The training and evaluation launchers are Bash scripts and require a Linux/POSIX environment and two suitable CUDA devices for a protocol-matched run. Installing dependencies alone does not download the dataset.

## Data preparation

Obtain the official U6G XL-MIMO Multi-config Radiomap dataset separately and provide its extracted `Dataset` directory. Data licenses and access rules remain with the dataset provider. The raw data and approximately 6.5 GB of processed arrays are not bundled.

```bash
python scripts/prepare_data.py --raw-root /path/to/Dataset
```

The script follows the frozen record manifests and creates `data/{train,val,test}_{condition,target,valid}.npy`, record lists, geometry files, and `input_audit.json`. The condition shape is `[N, 6, 256, 256]`; target shape is `[N, 1, 256, 256]`. Continuous dB maps use `(dB + 300) / 300`; invalid/building cells are masked. Consult [`data/README.md`](data/README.md) for the complete cache schema. Never use test targets for model selection.

## Inference with the archived checkpoint

Prepare an NPZ file containing `condition` with shape `[6, 256, 256]` or `[B, 6, 256, 256]` in the channel order above. The source-valid channel must be binary. Then run:

```bash
python scripts/infer_npz.py \
  --input /path/to/condition.npz \
  --output outputs/prediction.npz \
  --checkpoint checkpoints/t3_step5000.pt \
  --scene-id example-scene --draws 4 --nfe 8 --device cuda
```

The output contains normalized and dB-valued predictions. For an angle sweep of one scene, keep the 8 × 8 source channels unchanged and vary only the target beam channel. Do not describe an inference-only example as a full 160-scene test reproduction.

## Training and evaluation

The following commands preserve the archived entry points. They are **not** quick smoke tests and may require substantial GPU time and disk space.

```bash
# From the repository root, after data preparation:
NPROC=2 PYTHON=python bash scripts/run_training_from_scratch.sh
NPROC=2 PYTHON=python bash scripts/run_evaluation.sh val
NPROC=2 PYTHON=python bash scripts/run_evaluation.sh test
```

The training command writes a new run to `experiments/retrain_seed42/` and does not overwrite `checkpoints/t3_step5000.pt`. Validation must be used to select a checkpoint before the held-out test is evaluated. The provided `run_evaluation.sh` uses the archived T3 checkpoint compatibility path in `jobs/T3/`; it does **not** automatically evaluate a newly retrained checkpoint. Check the script arguments before changing any checkpoint path.

The archived formal test comprises 160 scenes × eight target angles, four draws, and Euler NFE 8. Training targets include −14°/0°/+14°; the remaining tested angles are interpolation or extrapolation, not an all-angle zero-shot training regime. Report metric grid and aggregation definitions: 256 × 256 model-grid and native 128 × 128 scores are not interchangeable.

## Integrity and reproducibility

`checkpoints/t3_step5000.pt` is the original checkpoint. Its expected SHA-256 is `611910c197d5a7a79a549654912ae7d637ac8bd030fb099a0bca69c43fc1cd97`. The source archive's original snapshot and checksums are retained under `provenance/ORIGINAL_*` and describe **the unedited archive**. Because this English-language copy changes text and project metadata, those source-file hashes cannot be presented as hashes of this tree. `CHECKSUMS.sha256` is regenerated for this copy.

The current `snapshot.json` records the edited source bytes for new training. With the author's approval, the distributed evaluator has a narrow compatibility rule: it accepts the original T3 source identity only when the checkpoint digest matches the archived checkpoint exactly. Other checkpoints must match the current snapshot. This permits evaluation without relabeling the archived checkpoint as newly trained code; it does not establish numerical equivalence of a fresh training run.

```bash
python scripts/verify_package.py
python scripts/verify_package.py --checksums
python scripts/verify_package.py --checksums-only  # No PyTorch import
```

The first command checks the archived checkpoint and strict model loading; the second also checks all files listed in the refreshed manifest. The last command checks only the current-copy files and can run without PyTorch. None runs a full scientific reproduction.

Before interpreting any paper table, record the exact checkpoint, scene split, target configurations, valid-pixel mask, number of stochastic draws, metric grid, and aggregation rule. An integrity check confirms file identity and model loading, not numerical reproduction of every reported result.

## Submission audit

The [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) require anonymous submissions and allow code as supplementary material. The manuscript's page limits and required AI-use statement do not define a README format. This local copy still contains historical path and license-attribution information from the source archive; those items require an author-approved blind-release decision. Do not upload this directory as anonymous supplementary code until a separate identity and dependency-license review has passed. No author name, affiliation, or repository URL is intentionally added by this README.

## License

The source archive includes an MIT `LICENSE`. Its attribution is preserved rather than silently changed for anonymous review. Dataset and checkpoint redistribution rights require separate confirmation.
