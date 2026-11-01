# B2 / M1 / M2 / M3: Paired 8x8 pilot and conditional 16x16 check

**Status:** execution protocol, not experimental results. Each group has one training run with seed 42. Sampling seeds are not substitutes for independent training seeds.

## Research scope

Test whether correlated random features, multiscale correlation lengths, and adaptive feedback in the U-NO encoding path improve radio-map detail prediction. The external flow-matching source remains x0 ~ N(0,I). This study does not validate a structured p0 and does not call auxiliary random fields observed electromagnetic phase.

A single 16x16 M3 run is permitted only if all four 8x8 groups train successfully and pass the validation gate. Failure or insufficient evidence does not trigger larger runs or post-hoc tuning.

## Paired design

| Group | Three innovation components at each encoder level | Correlation lengths across levels | Mixing weights |
|---|---|---|---|
| B2 | Independent standard white noise | Not applicable | Fixed at 1/3 |
| M1 | Zero-mean, unit-variance Matérn, ν=1 | [2,4,8]/256 of domain length at every level | Fixed at 1/3 |
| M2 | As above | [2,4,8]×2^s/256 at level s | Fixed at 1/3 |
| M3 | Same base innovations as M2 | Same as M2 | Spatial softmax from downsampled previous-level features, current condition encoding, and t |

“Single-scale M1” means correlation lengths **do not change across levels**. It still mixes the same three lengths to control the number of components. Stage placement, projections, initial gain 0.05, and decoder path are identical. All models contain the same-shaped head: frozen for B2/M1/M2 and trainable for M3. M2/M3 start with identical shared parameters and initial weights. M3 has more trainable parameters; this must be reported. Without a parameter-matched deterministic-feedback control, increased trainable capacity remains an alternative explanation.

Existing Tx/Height/Beam conditions, CA/SA, FNO modes (12,12,8,4,4), and operator width 32 are retained.

Matérn covariance is C(r)=(r/λ)K_1(r/λ), C(0)=1. Here λ is a kernel scale, not a fitted decorrelation distance. Lengths are fractions of the domain, not unverified meters; the earlier 20–40 m energy ratio is not imported. See the [official GaussianRandomFields.jl tutorial](https://pieterjanrobbe.github.io/GaussianRandomFields.jl/stable/tutorial/).

Use positive-definiteness-checked circulant embedding on a common extended grid. Filter real Gaussian noise by the square root of covariance eigenvalues, then crop the domain. Increase padding if eigenvalues are negative beyond rounding error; do not manufacture success by arbitrarily truncating the negative spectrum. Do not force sample-wise demeaning/normalization, which would change the Gaussian law and covariance.

Sample one independent innovation bank per ODE trajectory and reuse it across all Euler steps and CFG branches. Feedback weights may vary with t and state. For training, derive the bank independently and deterministically for each scene/epoch using a private RNG that does not consume the random streams of x0, t, CFG, or data ordering. Resume must reconstruct the same bank.

Independent-scale innovations are not strictly resolution-consistent observations of one field; cross-grid projection consistency is not tested. Once weights depend on random state, the final features are generally not jointly Gaussian and must not be described as a global GRF with analytic covariance. In ideal infinite-capacity FM risk, auxiliary noise independent of target and x0 could be ignored; any benefit is a finite-capacity inductive-bias hypothesis to be tested.

## Training and resources

Reuse the width32_b128 baseline: 6.7 GHz, 0 degrees, 560/80/160 scene split, training-only height normalization, and a 256² interpolated training grid. Original radio labels are 128²; the 256² grid does not add measured high-frequency information. Reuse the locked loader's valid-pixel and building/invalid exclusion rules.

Per group: global batch 128; two GPUs with synchronized, complex-safe gradient reduction; AdamW lr=1e-3, weight_decay=1e-5; EMA 0.999; AMP. Maximum 1,000 epochs / 5,000 updates; 500-step warmup; original 20-epoch early stop. Report actual budget differences if a group stops early. Select best EMA with the same per-epoch validation rule: N=2, CFG=1, fixed sampling seed.

The planned four-group parallel run uses eight idle A100s, two per group. Separate smoke checks must cover training, model/EMA agreement across ranks, checkpoint recovery, and the full validation set. Do not rewrite old width32 code or checkpoints. Freeze a code snapshot, manifest, package versions, and source hash; rank checkpoints retain their own RNG states.

## Evaluation and evidence

After training, freeze best EMA. Evaluate validation and test separately with Euler N={2,4,8,12} and four paired realizations (both external x0 and internal banks keyed by scene). Report global masked dB-RMSE/MAE, SSIM, NMSE, and raw out-of-range fraction. Per-scene outputs include RMSE, bias, raw RMSE, Haar-detail error, valid support count, and semivariogram.

Primary detail metric: apply the same 2x2 averaging to prediction and the existing benchmark target, then a four-level orthogonal Haar transform. Compute SSE divided by coefficient count for detail coefficients at the two finest levels with entirely valid support. This is a common-scale diagnostic of the interpolated benchmark, **not recovery of a new original 128² ground truth**. Exclude any filter support touching buildings/invalid pixels; do not run spectral analysis on sentinels.

Haar errors assess alignment of coefficient location, sign, and magnitude; increased generated energy is not a substitute for correct recovery. A semivariogram is a second-order difference statistic over valid pixel pairs, not proof of a particular physical path. Save numerical predictions for every scene at N=2 and all four realizations, enabling later selection by fixed visualization rules rather than attractive examples.

## Preregistered 16x16 gate (read-only 8x8 validation)

Require all four criteria simultaneously for M1→M2 (scale), M2→M3 (feedback), and B2→M3 (combined):

1. On the same 80 validation scenes, first average four realizations per scene; mean fine MSE at N=2 must improve by at least 2%.
2. Use 10,000 scene-paired bootstrap resamples with seed 20260909; the lower bound of the 95% CI for fine-MSE improvement must exceed zero.
3. Mean scene RMSE must worsen by no more than 0.1 dB at each of N=2 and N=8.
4. Mean scene absolute bias at N=2 must worsen by no more than 0.1 dB.

This fixed engineering expansion gate is not a mathematical theorem or a formal multiple-testing result. The CI covers scene variation, not training-seed variation. Passing warrants only one 16x16 transfer check; it does not prove multipath causality, an FNO-truncation mechanism, or generative-distribution calibration.

If any data are missing, a NaN occurs, training fails, or the gate fails, mark 16x16 skipped/blocked and record why; never substitute test for validation. Train only one 16x16 M3 run with seed 42 and unchanged hyperparameters. Do not search 16x16 hyperparameters. Comparisons with the old 16x16 baseline are historical references, not replacements for a contemporaneous ablation.

## Output locations and interpretation

The suite_status.json file in the new suite root tracks progress; gate.json records thresholds and paired differences. Each group directory contains config.json, metrics.csv, best.pt, last.pt, sampler_specification.json, training_runtime.json, and training logs. On completion, generate val_diagnostics.json, test_diagnostics.json, val_predictions, and test_predictions. Missing files mean incomplete work, not zero error or success.
