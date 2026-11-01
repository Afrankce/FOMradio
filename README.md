# FOMRadio: Operator Flow Matching for Radio Maps

This repository collects two distinct U6G research experiments:

* **T3:** an archived cross-array flow-matching model that requires a complete same-scene 8×8 reference radio map at inference.
* **A/C:** exploratory models without an auxiliary or target radio map at inference. A feeds the target beam map to the velocity network; C combines a separate environmental operator, analytic array excitation, and beam-conditioned Matérn noise.

The two protocols use different inputs and training targets. Their results must not be presented as an identical-input comparison. A/C results are exploratory: earlier results had already been inspected, and the user selected two training seeds after the original plan.

![C architecture: separate environment, excitation, and beam-conditioned source](assets/method_c.png)

## Dataset and split

The official [U6G XL-MIMO Multi-config Radiomap dataset](https://arxiv.org/abs/2603.06401) is obtained separately under its provider's terms. `splits/scene_ids_seed42.json` records the exact disjoint **560 training / 80 validation / 160 test scenes**. Raw maps and processed caches are not redistributed.

All A/C experiments here use **6.7 GHz**. Training and validation use 8×8 arrays and steering angles −14°, 0°, and +14°. The 160 held-out scenes are evaluated as follows:

| Test group | Target array | Steering angles | Queries |
|---|---:|---|---:|
| Seen configuration | 8×8 | −14°, 0°, +14° | 480 |
| Array transfer | 32×32 | −14°, 0°, +14° | 480 |
| Angle interpolation | 8×8 | −7°, +7° | 320 |
| Angle extrapolation | 8×8 | −28°, −21°, +21° | 480 |
| Array + angle transfer | 32×32 | −28°, −21°, −7°, +7°, +21° | 800 |

The archived T3 instead trains on **32×32 targets** at −14°/0°/+14°, with a real 8×8/0° reference map. Its manifests in `t3/manifests/` retain the original target and reference file identities; see `t3/provenance/ORIGINAL_PROTOCOL.md`.

## A/C test results

Per-image MAE in dB, averaged across four paired noise draws, the angles in each group, the 160 test scenes, and seeds 42 and 43. Euler NFE is 8. Lower is better. A and C use different information paths and C has additional parameters, so differences are system-level.

| Group | A MAE ↓ | C MAE ↓ |
|---|---:|---:|
| Seen configuration | 5.855 | 7.666 |
| Array transfer | 11.545 | 16.719 |
| Angle interpolation | 6.371 | 11.297 |
| Angle extrapolation | 6.421 | 11.534 |
| Array + angle transfer | 11.329 | 16.636 |

Full secondary metrics, per-seed results, scene-paired bootstrap intervals, and per-scene differences are in `results/ac_two_seeds/`. The bootstrap resamples scenes; it does not represent the full uncertainty across training seeds. These results do not establish cross-frequency or discretization-transfer generalization.

## Repository layout and availability

* `t3/`: archived T3 source, training/evaluation entry points, source-observation protocol and record manifests.
* `ac/`: A/C source, tests, test configuration, metric definitions and frozen exploratory plan.
* `splits/`: exact scene ID lists.
* `results/ac_two_seeds/`: verified aggregate test results and fixed-scene visualizations.

Model weights are **not included in this source snapshot**. The T3 checkpoint and A/C best checkpoints must be obtained separately from the authors; their absence prevents reproducing pretrained inference from this repository alone. The original T3 README discusses the complete local package and is qualified by a note in `t3/README.md`.

The A/C evaluation snapshot was run on Linux, PyTorch 2.5.1+cu121, NumPy 1.26.4, SciPy 1.13.1 and NVIDIA RTX 3090 GPUs. Set `U6G_DATASET_ROOT` to the extracted dataset for `ac/evaluate_test_ac.py`. The public copy substitutes that environment variable for the original machine-specific dataset path; the numerical model and metric operations are unchanged. Result metadata retain the hashes of the original experimental source files, not the edited release copy. The released scripts and missing weights still require an independent end-to-end reproduction check.

## Interpretation

C's four complex environmental bases are learned latent representations, not verified physical propagation paths or measured complex channels. Published beam maps match the checked LoS convention in typical pixels, but dataset metadata and codebook conventions have unresolved differences. No auxiliary or target radio map is an A/C inference input; target labels and validity masks are used for offline evaluation.

The release copy of `ac/plan_frozen.md` replaces one machine-specific path with `t3/`; its digest in `ac/test_run_spec.json` is updated accordingly. Original test records retain the original plan and source hashes, so they cannot be mixed with newly generated records as resumable units. The `raw/` directory contains individual test-metric records, not prediction images or radio maps.
