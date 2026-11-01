# T3: observed Y8 plus noise as the FM source

## Single changed factor

User-selected follow-up after T2 passed its validation gate. Only T3 is newly
trained; the completed T2 is the primary reference. Do not warm-start from T2,
change the velocity architecture, add nonstationary modules, or tune noise scale.

All fields below are in the existing normalized dB-map coordinates, NOT complex
electric fields, linear electromagnetic power, or physical propagation time.
Let c = [Tx, Height, Beam32, Beam8_0deg, Y8_0deg, source_valid]. Define mu8 as the
cached Y8 channel on source-valid cells and zero elsewhere. The unchanged noise
z follows the existing zero-mean unit-marginal-variance MS Matern sampler.

| Arm | Initial state | Conditions | Supervised velocity |
|---|---|---|---|
| T2 (completed reference) | z | c | Y32 - z |
| T3 (new) | mu8 + z | same c | Y32 - mu8 - z |

For T3, x_t=(1-t)(mu8+z)+tY32, t~Uniform[0,1]. Train the SAME velocity network
v_theta(x_t,t,c); integrate dx/dt=v_theta from x_0=mu8+z to t=1 at inference.
Noise scale sigma is exactly1 in these normalized coordinates, matching T2.
Do not clamp x0, intermediate states, residual targets, or final predictions.
Noise remains present at source-invalid pixels; only their source mean is zero.
The target valid mask is never used to form x0. No target-derived prior is added.

Writing r=x-mu8 gives r0=z, r1=Y32-mu8 and
r_t=(1-t)z+t(Y32-mu8). This is a conditional translation of residual FM.
Implementation evolves x directly with the unchanged network. It is not a new
physical scattering law or a guarantee of straighter learned ODE trajectories.

## Controlled data, training and evaluation

Reuse the exact parent T0/T2 caches and recorded hashes without modification.
Source is a real 8x8-array map fixed at0deg per scene; targets use the32x32 array,
6.7GHz. Angle coverage, source/target Tx/Rx settings, preprocessing and all masks
are unchanged. Scene split560/80/160; train angles -14/0/+14; evaluate eight.
The source remains fixed when the target angle changes. Real full source-map
availability is still assumed; this does not test predicted or sparse Y8.

Same SourceObservationUNO weights and parameter count at initialization as T2;
seed42 once, width32, original AP/NoDrop, online/noEMA, same fixed batch/noise/time
addresses, global batch128, microbatch16, two GPUs, AdamW lr0.001 wd1e-5,
500-step warmup/cosine,5000 successful updates. The velocity loss is unchanged:
masked FM MSE +0.1*(Haar1+Haar2)/2 after2x pooling. Its target changes exactly
as specified above; no new gradient or wavelet losses.

Previews1000/2500: same16 validation scenes/one draw. Final5000:80 validation
and160 test scenes/eight angles/four paired draws/Euler8. Compare final T3 to
final T2 and same-milestone previews only. Training losses use different velocity
targets and state distributions, so their magnitudes alone do not establish benefit.

Reuse T2 without retraining only after checking model/config/cache identities and
hashes. New training source and protocol are frozen in the new run. Parent results
are referenced, never edited. New NPZs record both base-noise and initial-state
hashes; base noise must match T2 for the same scene/draw, while initial states differ.

## Evidence and interpretation

Primary: unseen-angle deltaRMSE. Regional: native128 near-building RMSE and
shadow-transition gradient RMSE. Keep all map/native RMSE, shadow power, gradient,
Haar1-4 and morphology-orientation diagnostics from T2. No metric redefinition.
Geometry shadow/transition regions are proxies, not ray-tracing labels.

Freeze the same engineering thresholds, now relative to T2: at least1% response
improvement, at least3% native-near and transition-gradient improvement, all three
paired-scene95% upper bounds below0, region coverage at least20 scenes, and map
plus native RMSE at most1.01 times T2. Save this decision using validation before
T3 test evaluation. Report all test results even on gate failure. Bootstrap2000
paired scenes; one training seed, exploratory follow-up after observed T2 results.
No automatic next architecture, sigma sweep, or additional seed is scheduled.

Add a CPU-only deterministic reference that directly copies mu8 to all target
angles (no noise, no model). Use the identical target valid supports and explicitly
report source-missing coverage. Its angle response is zero by construction. This
is a copy baseline, not a calibrated beam-transfer or a stochastic-FM comparator.

## Execution and presentation

Only lease two currently idle GPUs from0-7 using shared locks. Preserve two-rank
training rather than occupying all GPUs for one arm. Persistent systemd service
and independent CPU monitor, SSH disconnect protection, alternate checkpoints
every250 successful updates, bounded retries, disk/stall checks.

Before formal training: unit tests for exact T2/T3 initialization, conditional
source masks, path endpoints/derivative, residual translation, Euler constant-
velocity endpoint, unchanged legacy source, and threshold logic; GPU short run;
2+2 versus direct4 resume check; paired underlying batch/noise/time audit against
T2; two-scene smoke sampling. Debug figures must remain separate from results.

Figure contract: real-data image plates and paired-effect plots answer whether
source translation improves target reconstruction AND target-configuration
response beyond T2. Reuse existing Python layout and field mapping, add only T3
category and source semantics. T2 reference is labelled as previously completed;
do not claim current training for it. Render PDF/SVG/PNG monitoring previews,
preserve prediction NPZ/JSON, and run alignment, glyph and collision QA. Per-arm
images retain the same hash-selected scene and shared physical colour limits.
