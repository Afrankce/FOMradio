# Beam-guided Matérn anchor ablation: seed 42

**Status:** exploratory one-seed engineering and validation run. Prior results and some test scenes have already been examined, so this is not a blinded or confirmatory study.

**Question:** under the A velocity model and its existing U6G split, how do finite noise-anchor sampling, Beam-guided anchor probabilities, and learned per-pixel scale weights affect radio-map generation without auxiliary radio maps?

**Inputs:** the existing 560/80/160 scene split and existing train/validation manifests. Training/validation: 6.7 GHz, 8×8, physical angles -14°, 0°, +14°. Read original data and original experiment artifacts without modifying them. Training labels are used for the loss; prediction takes Tx, height, target beam, and internally generated randomness, without target or auxiliary maps or target masks.

**Seed:** 42 only, including model initialization, minibatches, time, Gaussian fields, and anchor streams. Results cannot quantify training-seed variability.

**Conditions:** preserve the A velocity architecture and A training loss. Retain the original A baseline if its checkpoint and identity match. Train four matched variants: uniform-q/fixed-pi, Beam-q/fixed-pi, uniform-q/learned-pi, and Beam-q/learned-pi. All four use actual sampling with replacement; uniform q with finite anchors is a different source from original A.

**Initial sampling parameters:** 4096 anchors with replacement on the 256×256 grid, strength lambda=0.5, spatial probability uniform floor eta=0.2, fixed score alpha=beta=1 and temperature 1. All four variants use the same K, lambda, gradient normalization from the training split, source random stream conventions and variance normalization. These settings are exploratory starting values, not a claimed optimum. An independently drawn equal-scale MS field and the guided branch retain the A pairing coefficients sqrt(0.75), 0.5.

**Scale weights:** scales 2, 8, 32 pixels; a 355-parameter two-layer network receives target Beam and its normalized gradient, has an exactly uniform initial output, and has scale-floor 0.05. Freeze its parameters for fixed-pi groups; train them for learned-pi groups. Keep q parameters fixed in every group. The network may condition on Beam even when q is uniform.

**Execution:** first verify synthetic deterministic counts, source fallback, numerical variance, gradients, model/checkpoint handling, and the existing release tests. On GPU, start with a small real-data smoke test and inspect memory, losses, finite gradients, synchronization, and checkpoint resume. Then run 200 successful training updates for the four variants, using identical data/model initialization and a pilot microbatch appropriate to the available GPU. If checks pass, continue to 5000 successful updates with the same scheduler and validation/checkpoint schedule as A. Do not automatically change source settings or silently mix training protocols after seeing metrics.

**Evaluation:** choose checkpoints only by the frozen validation queries (240 queries, 8 Euler steps and one paired noise draw). Evaluate matched held-out test groups only after training and checkpoint selection are complete; report per-group metrics and single-seed status rather than uncertainty across seeds. Existing 6.7-GHz experiments do not establish cross-frequency transfer. Per-scene paired diagnostics are exploratory, not confirmatory.

**Safety:** use a new local/remote directory; do not modify or kill unrelated GPU jobs or paused generation. SSH credentials are entered locally and never saved; existing data and results remain untouched. Stop before launch if identity, available resources, checkpoint integrity or output isolation fails.
