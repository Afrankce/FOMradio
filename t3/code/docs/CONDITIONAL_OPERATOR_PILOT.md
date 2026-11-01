# Conditional operator pilot: locked 8x8 experiment

## Question and scope

Does explicit condition-dependent nonstationary feature transport improve CKM
structure over generic conditioning? This is a single-seed architecture pilot,
not a physics solver, novelty proof, or cross-array generalization result.

Four fresh runs: CASA (original), ADD (remove CA/SA but retain condition addition),
FILM (condition-dependent local affine modulation), NSOP (condition-dependent
input/output modulation around shared Fourier transport). All receive the same
Tx, height, beam and coordinate inputs through the unchanged condition encoder
and original input concatenation. No condition channel is removed.

## Locked controls

- 6.7 GHz, 8x8 array, 0-degree beam; existing scene split: 560/80/160.
- 1000 epochs, seed42, width32 U-NO, original modes/padding/scales/skips.
- Global batch128, two GPUs per run, original optimizer/scheduler/EMA/AMP.
- MS external Matérn source, independent linear sigma0 path, no internal noise.
- All groups: masked FM MSE + 0.1 times the same two-level Haar auxiliary loss.
- No feedback, wavelet network, dynamic K, extra geometry inputs, or path changes.
- Common layers initialized identically. FILM and NSOP have identical state-dict
  initial values and parameter counts. CASA and ADD differ in removed attention
  parameters. Actual counts and raw-gradient diagnostics are recorded.

## Exact replacements

CASA: original CA(h)*h + SA(h)*h + P(c).
ADD: h + P(c).
FILM/NSOP: start with ADD, then lift to width32. The same per-stage condition
MLP (1x1 embedding->32, GELU, 1x1 -> 2*J*32; J=2) yields a,b = 0.5*tanh(MLP(c)).
Last MLP weights have normal std0.01 and zero bias; private init seed314159.

FILM: h_tilde = (1 + mean_j a_j)*h + mean_j b_j, then original spectral/local update.
NSOP: K_c h = K h + mean_j [a_j * K((1+b_j)*h)]. Same K weights are used for
every term. Local branch and residual remain unchanged. Modulations are padded
only after construction; padded source support is zero. No hard building mask.

This is a restricted separable conditional-kernel hypothesis. It does not
guarantee directional ray tracing or high-frequency recovery. NSOP uses three
spectral calls versus one for FILM; equal parameters are NOT equal compute.
Wall time, GPU memory and epochs must be reported. No fabricated FLOP parity.

## Evidence and decision rule

Primary: epoch1000 last EMA, validation/test N8, scene-equal dB RMSE and fine
Haar MSE. Secondary: N2/4/12 sensitivity, orientation signed correlation,
coefficient mean squares, near-building and far-building errors, variograms,
raw clipping fractions, runtime, main/aux gradient norms/cosine by module.
Validation also includes N32. Four common source draws per scene are technical
repeats, not independent training seeds. Test is evaluated once at the locked
endpoint, never used for tuning or selecting weights.

An NSOP gain over CASA alone is insufficient: compare ADD and parameter-matched
FILM. If only global RMSE improves without fine/near-building/correlation gains,
do not claim improved propagation structure. Same-beam/same-array pilot cannot
support cross-beam response or unseen-array generalization; these remain next
stage gates. Source and path stay fixed even if the pilot underperforms.

## Verification, monitoring and figures

CPU tests: common initial weights; FILM/NSOP parameter parity; finite nonzero
condition/parameter gradients; zero modulation reduces to ADD. Real two-GPU
smoke for every variant: two updates, rank equality, save/reload, EMA, N2/N8 eval.
All smoke checks must pass before any full training starts. No automatic tuning.

Server systemd user services with linger run independently of SSH/client sleep.
Heartbeat30s: supervisor/worker presence, GPU/memory/temperature, stale progress,
disk space, AMP skips, render failure. Persist alerts to health.json and journal;
no chat push integration is available. Every100 committed epochs: training and
gradient statistics; full80-scene validation N2/N8 with four source draws and
predictions. Early four-scene previews at epochs1/10/50 use isolated RNG.

Plot contract: Python; existing diagnostic figure structure adapted from two
loss groups to four condition groups. All committed records retained; comparison
maps use common epochs and first four lexicographic validation scenes, never
error-selected. Separate curves show objective strength, parameter-gradient
alignment and validation error. Maps use shared ranges and invalid masks.
Report source CSVs and PNG/PDF/SVG with geometry QA; these are monitoring figures,
not evidence of a significant or physical gain. Figure uncertainty: one training
seed, descriptive epoch means, no cross-seed confidence intervals.

Code and manifests are snapshotted before execution. Each run keeps its own
checkpoints and resume identity; previous runs and models are not overwritten.
