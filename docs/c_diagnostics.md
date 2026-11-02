# Exploratory diagnosis of variant C

Variant C has higher test MAE than variant A in all five evaluated groups.
The A/C systems differ in their conditioning paths, source mean, parameters,
and loss, so these observations do not identify a single causal factor.

## Test-set localization

The following means use two training seeds, 160 held-out scenes, four paired
noise draws per query, and Euler NFE 8. MAE is measured per image over valid
pixels in dB and averaged over draws, angles, scenes, and seeds. The
deterministic mean `mu` is repeated across draws; the repeated values are not
independent observations.

| Evaluation group | A output | C mean `mu` | C output |
|---|---:|---:|---:|
| Seen configuration | 5.855 | 6.637 | 7.666 |
| Array transfer | 11.545 | 26.991 | 16.719 |
| Angle interpolation | 6.371 | 24.100 | 11.297 |
| Angle extrapolation | 6.421 | 30.569 | 11.534 |
| Joint array and angle | 11.329 | 40.012 | 16.636 |

The [per-group diagnostic](../results/ac_two_seeds/c_existing_diagnostic.json)
also reports per-seed, per-angle, boundary, and SSIM summaries. The results
are exploratory; the same test outcomes had already been inspected when this
diagnosis was designed.

## Excitation coverage

C predicts four complex basis maps from the scene. Its fixed 8×8 training
configurations use only −14°, 0°, and +14°. The resulting `3 × 4` complex
excitation matrix has singular values approximately `20.687, 17.223, 0.167`,
a nonzero-spectrum condition number of approximately 124, and at least one
complex null direction. Thus those configurations alone do not identify all
four basis directions. Power-only labels introduce additional ambiguity.

The relative distance of the 8×8 ±7° excitation vectors to the complex
training row span is approximately 0.184; for ±21° it is approximately 0.688.
These distances describe the implemented coefficient geometry; they do not
prove the cause of prediction error. See the
[excitation diagnostic](../results/ac_two_seeds/c_excitation_diagnostic.json).

## Beam-to-noise intervention

For each of two trained C checkpoints, six validation scenes were selected
by scene ID and evaluated at −14°, 0°, +14°, −7°, and +7° using one paired
draw. The scene condition, array configuration, structured mean, checkpoint,
and base random draws were held fixed. Only the beam input to the noise
probability map was replaced with a uniform map or with a different-angle
beam. Target labels were read after predictions were generated.

| Diagnostic | Seed 42 | Seed 43 |
|---|---:|---:|
| Correct-beam MAE (dB) | 8.655 | 8.748 |
| Uniform-q MAE (dB) | 8.672 | 8.761 |
| Prediction change, uniform versus correct (dB) | 0.141 | 0.148 |
| Prediction change, wrong versus correct beam (dB) | 0.195 | 0.209 |

The [per-query validation records](../results/ac_two_seeds/c_noise_validation_probe.json)
also contain trajectory and noise-probability diagnostics. These are
inference-time interventions with a small validation sample and no 32×32
queries; they are not retrained ablations. The outcomes show a small effect in
this probe rather than general ineffectiveness of the noise branch.

In C the target beam is not a direct velocity condition. Its noise branch
primarily modifies spatial covariance after pointwise variance normalization;
the independent and beam-guided components have weights `sqrt(0.75)` and
`0.5`. Whether this weakens the learnable beam signal should be tested using
matched retraining.

## Source scale and next checks

The source noise has unit standard deviation in `(dB + 300) / 300` coordinates.
On the same validation probe, the mean-state MAE averaged 13.05 dB across
both seeds and 30 queries, while the noisy initial state averaged 232.24 dB
after conversion to the same linear dB scale. These intermediate flow states
are latent variables, so their numerical range alone is not a physical error.
A uses the same noise scale, and no causal comparison has isolated this factor.

The next tests are matched retraining with direct beam conditioning retained
or removed, the structured mean and beam-guided noise added separately, and
alternative source scales. Training and evaluation splits, budgets, and
seeds should be held fixed across arms. Changes in basis count or training
angles should be studied separately, alongside excitation singular values and
held-out configuration coverage.
