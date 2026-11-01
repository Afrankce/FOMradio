# MS-source L1 / L2 detail-loss pilot

## Locked scope

8x8 array, 6.7 GHz, 0-degree beam, scene split and preprocessing copied from
the completed external-source suite. 560 train / 80 validation / 160 test.
MS source, no internal noise, unchanged width32 U-NO / conditions / linear path.
Two GPUs per group, batch 128, seed 42, 1000 epochs, same optimizer and EMA.
Final checkpoint is epoch1000 last EMA, not best-checkpoint selection.
L0 is the existing MS run; do not retrain or modify it. Compare matched epochs
and inference steps; diagnostic repeats are not independent training seeds.

## Objectives

Let e = predicted velocity - (target - source), in normalized coordinates.
FM is squared error averaged over globally valid pixels.
Auxiliary objectives use the shared 2x2 average of e (256 -> 128 benchmark
proxy, not restored native observations). Only complete filter supports count.

- L1: mean squared horizontal/vertical neighboring differences on that grid.
  L1 names the variant; it does NOT mean absolute-value/L1-norm loss.
- L2: equal-weight mean of the first two Haar detail-band MSEs, each averaged
  over all three orientations and its own globally valid support count.
- Total = FM + 0.1 auxiliary. The coefficient is fixed, untuned, no search.

Invalid values are removed before transforms; no zero-filled boundary enters
valid coefficient supervision. No clamp on velocity and no dB multiplier.
Test data and test metrics never choose weights or stopping thresholds.

## Diagnostics

Every optimizer window: FM, gradient, Haar1, Haar2, weighted auxiliary, total,
support counts, AMP scale/skips, output-gradient norm/ratio/cosine.
Every 25 successful optimizer steps (also first): separately differentiate FM
and unweighted auxiliary, SUM-reduce gradients across ranks before norms or
cosines, use real-coordinate complex gradients and remove AMP scaling.
Report all parameters and spectral/local/attention/condition-encoder/other.
These are raw gradients before clipping/Adam, not physical spatial gradients
and not realized Adam update vectors. Diagnostics do not consume random draws.
The legacy train_loss column remains FM MSE; mixed total is separately logged.

Every epoch: validation N2, checkpoint, resume marker. At epochs 1/10/50:
four fixed validation scenes, N2/N8 previews. Every100 epochs: all80 validation
scenes, four source draws, N2/N8 maps and masked wavelet/gradient/variogram metrics.
At completion: validation N2/4/8/12/32 and test N2/4/8/12, four draws.

## Hypothesis and physical interpretation

Question: does explicitly weighting local differences improve the alignment
and contrast of environment-dependent spatial structures, not only texture?
Haar coefficients describe signed spatial variation of a log-power map, not
RF phase, electromagnetic energy, or separately identified reflected rays.
Analyze scale/orientation errors, near-building vs far errors, signed bias,
and complete-support coverage. Building distance uses height>0, not target
power. Absence of ray-level truth precludes assigning a subband to reflection
or diffraction. Global benefit and local failures must both be reported.
Lower auxiliary loss alone is not evidence of better generated endpoints.

## Monitoring and visualization contract

Server-side lingering systemd supervisor and monitor, independent of SSH.
30-second heartbeat; missing process, stale training progress, low disk,
failed rendering and AMP instability are persisted as alerts. No automatic
hyperparameter changes or new experiment groups. Chat push is unavailable.
Python reporting figures (not journal submission figures): loss allocation,
parameter gradient strength/conflict, matched-epoch validation, fixed-scene
GT/prediction/error. JSON/CSV source data and PNG/PDF/SVG exports.
No confidence intervals across steps or source draws presented as seed CIs.
