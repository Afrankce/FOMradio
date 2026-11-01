# Conditional spectral gate: registered pilot

Status: execution protocol, not results. Four experiments only: B, R, A, AP.
8×8 is the antenna array, not map size or retained modes.

Frozen controls: 6.7 GHz / 0-degree beam; same 560/80/160 scene split; MS Matérn
source and linear independent FM path; no internal random fields; CASA fusion;
width32; modes(12,12,8,4,4); existing padding; global batch128 over2 GPUs;
1000epochs, seed42, original optimizer/LR/warmup; FM +0.1 Haar objective.
Online weights only, compatibility ema fields alias online weights. No checkpoint
or solver selection using test. Main final comparison N8; N4 sensitivity.
Every100epochs: full validation N2/N8 ×4 fixed source draws, time diagnostics,
raw saved predictions, gate probe on one fixed validation scene along N8 flow.
Early fixed-map previews at epochs1/10/50. Per-epoch validation is N2/draw0 and
must not be confused with the registered N8 endpoint. Gradients every25updates.

R: bounded radial rebalancing 1+0.5*tanh(raw)*rho^0.7, raw=0 initially,
rho=sqrt(fx²+fy²)/sqrt(0.5), cycles/padded-grid-pixel. This is a bounded,
DiffFNO-inspired control, not a full DiffFNO reproduction. Gamma may be negative.
A: log normalized magnitudes of state/condition, signed normalized ky/kx and
learned time features -> MLP32 -> real amplitude gate in [0.5,1.5].
AP: adds normalized relative cross-spectrum real/imag inputs plus bounded phase
output in [-pi/4,pi/4]. Both heads zero initialized. Separate weights per stage.
For all variants kx=0 and Nyquist columns remain unchanged. This preserves
legacy rFFT boundary handling and makes the new residual spectrum Hermitian-
compatible; no new claim of full-model translation equivariance. No K expansion.
Raw field Fourier phase is spatial spectral phase, not RF carrier phase.

A vs AP changes both descriptors and output type. Follow-up isolation is required
before attributing gains specifically to phase rotation. No follow-up experiments
or16×16 jobs are automatically launched in this suite. Parameter counts/costs
are reported, not claimed equal. One seed is exploratory; repeated source draws
are not independent training seeds. Known test has been previously explored.

Prerequisites: all common weights and caller RNG identical; identity output;
finite complex/backprop gradients; valid real spectral increments; two-rank
synchronization, checkpoint resume roundtrip, online/no-EMA identity and smoke
validation. All four pass before full training. Failures stop affected jobs;
no automatic parameter tuning, data changes, checkpoint rewriting, or OOM batch
reduction. Resume only complete committed checkpoints of this exact snapshot.

Monitoring: independent server-side systemd service, 30-second health checks,
process/disk/stall/NaN/skipped-step/rendering alerts;100epoch JSON statistics.
No client connection needed while server remains on. Server files/journal are
the notification destination; no external chat-push transport is configured.
Visualization uses committed epochs and saved predictions, no independent GPU
inference. PNG/PDF/SVG and CSV; same color limits, valid supports, fixed scenes.
User-selected radiomap palette: jet; invalid/building black; fixed [-300,0]dB
across GT and predictions. Error maps retain magma [0,100]dB. Colors only, no
per-image rescaling or training-data changes.

Main interpretation: energy-only gain is insufficient. Check near-building
errors, signed Haar details, amplitude/phase spectral error on valid patches,
and module gradient magnitude/alignment. Local window FFT/wavelet, source,
path, loss, and fusion removal are separate later questions.
