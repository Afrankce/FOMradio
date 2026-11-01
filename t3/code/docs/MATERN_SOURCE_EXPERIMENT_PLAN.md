# External multiscale Matérn source: Experiment design for a fixed U-NO

Date: 2026-09-09. **Status: design/planned; not implemented and no training launched.** The original plan used the ccf-experiment-designer skill. This is a mechanism ablation, not a complete paper benchmark package. It claims neither novelty nor suitability for a particular conference.

## 1. Research question and scope

Primary question: With the same conditional U-NO velocity architecture, training budget, and evaluation NFE, does replacing the external source x0 ~ N(0,I) with a multiscale Matérn Gaussian source improve radio-map detail and spatial-correlation prediction?

Only source covariance changes. A fixed architecture does not mean frozen weights: each group trains independently from the same initialization. Changing only the inference source of a white-noise-trained checkpoint is invalid.

Disable all internal auxiliary random fields and use AttentionMultiscaleUNO2d(operator_width=32) directly; do not merely set the old M3 gain to zero. The model must have no InnovationInjection, innovation bank, feedback head, or source-variant embedding. Keep Tx/Height/Beam, CA/SA, time encoding, spectral truncation, up/downsampling, masks, and velocity MSE unchanged. Do not add a Beam mean, conditional covariance, SIREN, wavelet module, frequency-weighted loss, OT pairing, or learned source weights: those would test multiple mechanisms at once.

Old B2 is not the white-noise baseline for this study because it also injected internal white noise. Old checkpoints are historical references only, not fair primary-table comparators.

## 2. Claim–evidence matrix

| Hypothesis | Required comparison | Main evidence | Unsupported conclusion |
|---|---|---|---|
| A spatially correlated source helps | Each Matérn source vs W | Same-NFE detail/global errors and training stability | Internal noise modules are effective |
| Multiscale covariance adds value | MS vs all prespecified single scales, especially validation-selected S* | High/low-frequency errors, global guardrails, repeated training seeds | Every possible single kernel is inferior |
| Correct propagation detail improves | Sign/location-sensitive Haar errors, gradient errors, paired fixed crops | Lower coefficient error against the actual target | Matching PSD energy proves correct reflection locations |
| Improvement is not solely a low-step bias | Fixed checkpoint, NFE sweep, finer-step check | N=2/4/8/12 and validation N=32 | A low-step gain proves the exact flow is better |
| Predictive distribution improves, not merely smoothness | Single-sample and ensemble-mean metrics plus proper scores | CRPS/energy score, spread/error, correlation structure | One label per condition proves full conditional calibration |

## 3. Five main groups

All sources have zero mean and unit marginal variance on the same 256² grid. Original radio labels are 128²; the 256² target remains the interpolated benchmark and is not newly measured physical high-frequency data.

For ν=1, retain the project's kernel-length convention:

$$K_\lambda(p,q)=\begin{cases}(r/\lambda)K_1(r/\lambda),&r>0,\\1,&r=0,\end{cases}\quad r=\|p-q\|_2.$$

Both r and λ are measured in **pixels of the 256² source grid**. Do not confuse them with pixels of a downsampled U-NO stage, a 20–40 m wavelet scale, or decorrelation distance. The formula and sampling principles follow the [official GaussianRandomFields.jl tutorial](https://pieterjanrobbe.github.io/GaussianRandomFields.jl/stable/tutorial/).

| ID | Source distribution | Purpose |
|---|---|---|
| W | N(0,I) | Concurrent white-noise baseline without internal noise |
| S2 | N(0,K₂) | Shorter correlation length |
| S8 | N(0,K₈) | Intermediate correlation length |
| S32 | N(0,K₃₂) | Longer correlation length |
| MS | N(0,(K₂+K₈+K₃₂)/3) | Multiscale covariance, primary candidate |

Lengths [2,8,32] are prespecified at fourfold intervals to distinguish mechanisms; they are not data-fitted optimal scales. Do not use [2,4,8] as the only main comparator, which would explore a narrow range. Fix ν=1 and equal weights; do not search ν or mixture proportions in the first round.

Mathematically,

$$x_0^{MS}=\frac1{\sqrt3}(z_2+z_8+z_{32}),\qquad z_\lambda\sim N(0,K_\lambda)\ \text{independently}.$$

The marginal variance at each point remains one. This is a **Gaussian covariance sum**, not a Gaussian mixture made by drawing one scale at random. Its kernel is generally not a single-λ Matérn kernel, but it remains a Gaussian field.

Only one combined x0 tensor of shape [B,1,256,256] enters the velocity network. Do not stack three sources as extra channels, inject them by stage, or change parameter count. Multiscale structure belongs to the covariance of one external random function, not to new network branches.

Even if MS beats the three specified single scales, an untested single kernel at another effective length could do better. A broader paper claim would need a prespecified, validation-selected single-kernel search budget. This experiment supports only comparison with the specified candidates.

## 4. Entry into the actual flow-matching path

Training uses

$$x_0\sim p_0^{(a)},\quad (x_1,C)\sim p_{data},\quad t\sim U(0,1),$$
$$x_t=(1-t)x_0+t x_1,\qquad u_t=x_1-x_0,$$
$$\mathcal L=\frac{\sum m\,[v_\theta(x_t,t,C)-u_t]^2}{\sum m}.$$

Pair source and target independently. The source is unconditional and zero-mean. ConditionalFlowMatcher(sigma=0) retains the same equations, but **the sampled x0 and path distribution genuinely change**. Do not replace only the inference initialization, or sample Matérn noise while still calculating the supervised velocity against white noise.

For fixed x1 and t<1, the conditional path is

$$p_t(x\mid x_1,C)=N(t x_1,(1-t)^2K_a).$$

At t=1 it degenerates to the target point. The full marginal path also averages over data endpoints and is not a single Gaussian. This equation shows that source covariance persists along the supervised path rather than acting as an unrelated perturbation.

Inference draws x0 once from the same source specification and integrates fixed-step Euler from 0 to 1. There is no innovation bank and no intermediate noise redraw.

Interface sketch, **not yet implemented**:

~~~python
model = AttentionMultiscaleUNO2d(operator_width=32)
x0 = source.sample(base_noise)      # Sole experimental variable
t = time_sampler(batch_size)
xt = (1 - t) * x0 + t * target
ut = target - x0
vt = model(image=condition, x=xt, step=t)
loss = masked_velocity_mse(vt, ut, valid_mask)

# Evaluation uses the same source definition as training.
x0 = source.sample(eval_base_noise)
prediction = euler_cfg_sample(model, condition, x0, cfg_scale=1, steps=N)
~~~

## 5. Source sampling and fairness contract

1. Use positive-definiteness-checked circulant embedding, a shared extended grid, and a fixed crop ROI. Check eigenvalues for every prespecified λ. If the minimum is below -1e-10, enlarge the embedding; fail and block the study after maximum side length 2048 instead of silently changing covariance.
2. Filter real Gaussian noise with the **square root** of covariance eigenvalues. Zero mean means ensemble mean, not per-image demeaning; unit variance means ensemble marginal variance, not per-image standardization. Do not clamp sources, remove DC, or multiply sources by a building mask.
3. Use common random numbers: the same seed, scene, and epoch/realization select one real white-noise ε on the shared extended grid. Groups change only the fixed spectral filter. W directly crops ε.
4. For efficiency, MS may filter one ε with sqrt((eig₂+eig₈+eig₃₂)/3). This has the **same distribution** as the sum of three independent fields, changing only cross-group pairing. Filtering one ε with (sqrt(eig₂)+sqrt(eig₈)+sqrt(eig₃₂))/sqrt(3) is wrong because it creates cross terms.
5. Isolate RNG streams for model initialization, loader, t, CFG dropout, and source. Save the shared initial state_dict checksum for each training seed. Different source samplers must not alter t or training-data order.
6. Generate sources in FP32 and retain existing FP32 FFT behavior. Record source-spectrum dynamic range, numerical zero mode, memory, cache, sampling time, and minimum eigenvalue. Report both cold construction and cached sampling costs for MS; do not hide them outside inference timing.
7. Train W/S2/S8/S32/MS from scratch without transfer from old M3. All groups have identical trainable parameter count and forward graph.
8. Do not add an εI/nugget white-noise fallback. Any future test of one must be a separately recorded group, not an undocumented response to training failure.

## 6. Data, training, and checkpoint rules

- The first round uses only the 8x8 array, 6.7 GHz, and the 0-degree beam, with the existing 560/80/160 scene split, same manifest, and training-only height statistics. Save new input summaries; do not alter raw labels.
- Test scenes were viewed in earlier analyses. Even without further test-driven tuning, this study is subsequent exploration on an already observed fixed holdout, **not** a newly unseen confirmatory test. Stronger claims require unused configurations/data or external validation.
- Keep state channels (32,64,128,256,256), operator width 32, modes (12,12,8,4,4), padding (9,5,3,2,1), CA/SA, condition encoder, and skips unchanged.
- Per group: 1,000 epochs; five planned update windows per epoch; global batch 128; AdamW lr=1e-3 and weight_decay=1e-5; 500-step warmup; same scheduler; EMA=0.999; existing AMP; CFG dropout=0.25.
- No validation-based early stopping: fix the main comparison budget. Record all 5,000 planned windows, actual successful updates, and AMP skips. A failed-update rate over 1% is a numerical-stability anomaly. Do not treat a materially mismatched budget as a fair performance comparison; version a common numerical correction and rerun the affected full comparison.
- **Primary checkpoint is EMA at epoch 1,000**, avoiding selection of a smoother checkpoint by N=2 RMSE. Record normal validation curves each epoch. Every 100 epochs, compute N=2 and N=8 detail statistics on all 80 validation scenes with four realizations. Intermediate checkpoints are learning-curve diagnostics, not candidates for the primary table.
- Pilot: five groups × seed 42 = five training runs. Validation alone determines progression. Confirmation: add seeds 43 and 44 to **all five** groups, for 15 total runs; do not add seeds only to a winner.
- Recommended allocation is two GPUs and batch 128 per run. If eight GPUs are idle, four groups can run concurrently and the fifth can queue. This is resource planning, not authorization to launch or a guarantee of availability. Record GPU model, concurrent job count, wall time, and GPU-hours.

## 7. Evaluation: correct detail before attractive texture

### Primary endpoint and sampling budget

- **Primary endpoint:** finest-two-level Haar MSE for N=8 single-sample predictions. Average four realization-level metrics within each scene, then average scenes. List training seeds separately in confirmation; do not treat pixels or realizations as independent training replicates.
- **Primary comparisons:** MS vs W and MS vs S*. Choose S* among S2/S8/S32 using only pilot-validation N=8 fine MSE; ties favor shorter λ. Lock S* before extra seeds and holdout evaluation, while still reporting all three single-scale groups.
- N=2 is a secondary deployment endpoint and global-error guardrail; N=4 and N=12 form the curve. All groups use CFG=1 and identical NFE, without group-specific solvers or step counts.
- After freezing final EMA, additionally evaluate all five groups on validation at N=32 with four realizations to check whether a gain is restricted to few steps. N=32 is not automatically ground truth or an exact ODE solution. If 12→32 still changes materially, report unresolved numerical interpretation.

### Core metrics

1. Masked dB-RMSE/MAE, bias, SSIM, and raw out-of-range fraction. Report both clipped and raw RMSE to detect hidden instability; distinguish mean-of-scene primary statistics from pooled-pixel summaries.
2. Reuse scene_diagnostics fine MSE: apply the same 2x2 averaging to prediction and the current 256² target, then Haar; compute SSE/count over fully valid detail coefficients at the two finest levels. Preserve coefficient signs, not just energy.
3. Report errors at all four Haar levels and horizontal/vertical gradient errors on fixed valid neighborhoods. Insufficient fully valid support is missing, not zero.
4. Compute separate horizontal and vertical semivariogram errors at lags {2,4,8,16} benchmark pixels. Output prediction/target values, valid pixel-pair counts, and differences. The old function pooled directions; add new direction-specific fields without silently changing old ones.
5. Use spectral error on valid local patches only as a diagnostic. Predefine and save the window, demeaning, and frequency bands. Do not call an FFT of the full image after masking invalid pixels to zero a true propagation spectrum. Spectral phase means **spatial spectral phase**, not RF phase.

### Generative distribution versus point prediction

- The primary table reports **mean of single-sample metrics**. Put ensemble-mean prediction metrics in a separate table. Do not call the smoothing benefit of averaging four maps a single-sample generative improvement. Best-of-K oracle results are prohibited as primary results.
- In confirmation at N=8, generate 16 independent realizations for each training seed and all holdout scenes, with the prespecified first four included. Under a fixed budget, calculate valid-pixel CRPS, whole-map energy score using valid-pixel-normalized L2, and ensemble-mean RMSE.
- Use a finite-sample proper-score U-statistic pair term omitting i=j. For example, energy score is mean_i d(X_i,y) - sum_{i!=j} d(X_i,X_j)/(2K(K-1)). Do not clamp slightly negative estimates to zero; record K=16 and the estimator.
- Report sample spread, error, and interval coverage together. One label per heterogeneous condition gives limited evidence about aggregate predictive distributions, not proof of complete multipath uncertainty under any one condition. Coverage alone cannot be passed by merely inflating noise.

## 8. Progression gate and statistical rules

The thresholds below are **proposed engineering criteria**, not accepted validity theorems. Lock them before approving execution.

On pilot validation at N=8, four realizations per scene, and fixed final EMA, require:

1. At least a 2% reduction in mean scene fine MSE for MS relative to W.
2. At least a 2% reduction in mean scene fine MSE for MS relative to S*; otherwise do not claim an extra benefit over single-scale sources.
3. For both comparisons, mean scene RMSE must worsen by at most 0.1 dB at each of N=2 and N=8, and mean scene absolute bias by at most 0.1 dB.
4. Stable direction of paired fine-MSE differences: keep methods/realizations paired within each scene and run 10,000 scene-bootstrap resamples with seed 20260909. Pilot CIs describe scene uncertainty only. Because S* is selected on the same validation set, its comparison CI is not an independent confirmatory significance result.

Recommend adding two training seeds only if both effects and guardrails pass and both descriptive CI lower bounds exceed zero. Otherwise do not search for a favorable result; report conclusions separately for white noise, correlated sources, and multiscale covariance. Near-threshold results may prompt a future proposal, but the gate must not be changed silently.

Confirmation reports all five groups × three seeds. If p-values are supplied for the two primary comparisons, use scene-level paired tests and Holm correction. Report effect direction per training seed, paired-scene CIs, and sensitivity intervals resampling both scene and seed dimensions. Three training seeds still do not precisely characterize all training variation; do not pretend 3×160 are 480 independent training replicates.

Because the historical holdout was observed, external validity remains limited even if confirmation trends are stable. A future 16x16 study should run W, locked S*, and MS together rather than comparing one MS run with different historical models. That expansion is **not authorized** by this plan.

## 9. Mechanisms and failure classes

| Observed pattern | Plausible interpretation | Do not infer |
|---|---|---|
| MS beats W but not S* | Correlated sources may help; no extra multiscale evidence | Multiscale mechanism verified |
| MS beats W and S* with robust detail/global error | Preset multiscale covariance helps under this fixed model/budget | Source distribution encodes physical propagation laws |
| Gain only at N=2, absent at N=8/32 | Few-step generation better matches this source | Exact transport distribution must be superior |
| Closer PSD but worse Haar/gradient error | Texture energy is closer; position/sign may be wrong | High-frequency propagation recovered |
| Ensemble mean improves but single samples/proper scores worsen | Averaging cancels stochastic error | Generative distribution is more accurate |
| All smoother sources worsen | This source/path/model combination may be unfavorable, or high-frequency source support too weak | Matérn fails for every FM problem |

To distinguish smoother sources from better detail recovery, record fixed-validation velocity errors every 100 epochs in time bins t∈[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]. Construct x_t/u_t using each group's legitimate source with the same t, scene, and base randomness; save target-velocity variance as well. Do not compare training losses from different path distributions and assert superior optimization directly.

Multikernel Matérn covariance remains mainly low-pass and does not guarantee predictable high-frequency structure. Band-pass sources, conditional means, and anisotropic kernels are separate future factors. This source is a finite-grid Gaussian-function design; this round does not claim cross-resolution consistency or a complete infinite-dimensional OFM theory.

## 10. Pre-implementation checks and code boundary

Only isolated implementations are planned; do not alter the old suite snapshot:

- New source sampler: common extended grid, frozen source specification, covariance sum, private RNG, cache, and provenance.
- New source trainer: replace x0 generation at training/multiconfig_trainer.py:368; explicitly share time-sampling rules to avoid coupling torchcfm's internal RNG to source RNG; keep masked velocity MSE.
- New evaluation wrapper: use the U-NO without internal noise and the identical source definition; do not call predict_frozen or use_innovations.
- Per-group outputs: configuration/model hash, source hash, checkpoint, complete metrics.csv, 100-epoch statistics, per-scene/seed/NFE metrics, and rule-based visualizations.

Low-cost tests required before execution:

1. All five groups have equal parameter counts and initialization checksums. Without innovation modules or random sampling, evaluation-mode velocity forward is reproducible.
2. Kernel FFT/inverse FFT agrees with analytic C(0) and specified-lag covariance; MS is the arithmetic mean of three kernels. Monte Carlo mean, variance, and correlation match within prespecified sampling error.
3. W is genuinely iid. Real-valued/Hermitian, PSD, and crop-boundary checks pass. Similar-looking white-noise/Matérn images are insufficient evidence.
4. xt(t=0)=x0, xt(t=1)=target, ut=target-x0, and the finite-difference path derivative agree. Training and inference source specifications/hashes match.
5. Sample x0 only once per inference trajectory. Source sampling does not alter model/t/CFG/loader RNG; resume reconstructs the same subsequent random sequence.
6. A small train/validate/save/resume smoke run passes. Changing source and time actually changes model inputs, not only configuration labels.

## 11. Result template and execution order

| Source | Training seeds | NFE | Fine MSE ↓ | dB RMSE ↓ | Gradient MSE ↓ | Variogram error ↓ | Raw out-of-range ↓ | GPU-hours |
|---|---|---|---|---|---|---|---|---|
| W | 42 (43/44 in confirmation) | 8 | TBD | TBD | TBD | TBD | TBD | TBD |
| S2 | Same | 8 | TBD | TBD | TBD | TBD | TBD | TBD |
| S8 | Same | 8 | TBD | TBD | TBD | TBD | TBD | TBD |
| S32 | Same | 8 | TBD | TBD | TBD | TBD | TBD | TBD |
| MS | Same | 8 | TBD | TBD | TBD | TBD | TBD | TBD |

Planned figures: source examples with analytic/empirical covariance and spectra; training progress and time-bin errors; NFE–RMSE/fine-MSE curves; paired same-scene GT/prediction/error maps and fixed crops across five groups; Haar coefficients by level; separate single-sample, ensemble-mean, and proper-score views. Fixed scenes include previously discussed u799 and examples chosen by building-coverage quantiles, not by model advantage.

Execution order: source sampler and path-contract tests → five-group smoke runs → full-budget 8x8 seed-42 pilot → validation gate → if passed, add seeds 43/44 to **all** groups → lock comparisons and evaluate the historical holdout → decide whether an external check on a new configuration is needed.

Outstanding items: the new source sampler/trainer are not implemented; all five result rows are TBD; lengths have not been converted to verified meter scales; no submission venue was selected or current-method benchmark search performed; no completely new unseen test set is available.

This document **only designs an experiment**. It produced no training results and started no GPU jobs. Implementation and execution under this protocol require a user request.
