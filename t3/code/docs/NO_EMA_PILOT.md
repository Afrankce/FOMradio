# No-EMA matched-training and paired-weight evaluation

## Central question

Does EMA lag explain poor CKM reconstruction or the ranking of conditioning
modules? No architecture, source, loss, optimizer, batch, path or seed is changed.
The reference is conditional_operator_ms_8x8_20260910T093400Z.

First evaluate the model slot of each reference last.pt on validation and test,
using the same source draws and N=2/4/8/12 (also N32 validation). Compare with
the already archived reference EMA evaluations. Record reference file hashes,
parameter hashes and identity. No model selection uses these new test results.

Then train CASA/ADD/FILM/NSOP from the same initial states, each on two GPUs,
1000 epochs, global batch128, seed42, MS source, linear sigma0 path, FM + 0.1 Haar.
This is standard gradient training without weight averaging, NOT a new FM-only
loss ablation. Validation, previews, milestones and final test use online weights.
Historical milestones only contain EMA weights, so a fresh trajectory is needed
to collect online checkpoints at every100 epochs. Endpoints are fixed, not best.

NoEMA uses a no-op adapter to the inherited trainer interface. The ema_model
attribute aliases the online model; there is no second maintained model, no
average, and no freeze of online parameters. For schema compatibility the
checkpoint ema slot aliases the model slot and MUST equal it. Config explicitly
records ema_enabled=false, ema_decay=null, evaluation_weights=online. Milestone
file ema.pt is a legacy filename holding ONLINE weights in this suite only.
This convention must not be mixed with reference milestone ema.pt files.

Online training should be invariant to removing the passive EMA update, since
EMA never enters loss and early stopping is disabled. Test this numerically:
CPU optimizer trajectories, reference-versus-new two-step smoke online hashes,
and final online hashes; report rather than hide any divergence. Best checkpoint
times can differ but never determine the final comparison. Small GPU numerical
differences, if observed, must be separated from exact paired reference-weight
evaluation. Initial models, manifests and noise streams are verified.

Primary evidence: scene-average RMSE and finest-two Haar MSE at N8; secondary
signed coefficient correlation, near/far-building error, variograms, clipping,
NFE sensitivity, FM/Haar losses and raw parameter-gradient ratios/cosines.
One training seed, 80 validation scenes and160 test scenes; four source draws
are technical repeats, not independent training seeds. No significance claim.

Decision: online better early but EMA catches up suggests lag; same spatial
errors in both argues against EMA being the main bottleneck. No decay sweep,
new architecture, extra conditions, or adaptive hyperparameter search this round.

All four pairs must pass smoke before training. Frozen code/manifests, separate
run roots, safe resume, durable systemd services with user linger. Heartbeat30s,
statistics and full validation maps every100 epochs, isolated early previews.
No chat push or server-reboot recovery is claimed. Completed old files read-only.

Reporting plots inherit the established Python monitoring workflow, with ONLINE
weight labels and fixed shared ranges. Source CSV, PNG/PDF/SVG and geometry QA.
No fabricated results, no cross-seed uncertainty, no inference about RF phase.
