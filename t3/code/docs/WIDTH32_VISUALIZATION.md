# Visualization of the width32 experiments

Automated export covers the three formal experiments (8x8, 16x16, and 32x32) and the batch-64 rerun after the out-of-memory event.

To preserve the source fingerprint of training runs already in progress, export uses a separate sidecar and does not modify the training Python files. Its program, configuration, validation records, and instructions are at:

[/home/ka/CKM/radio-unet/runs/width32_b128_20260908T052359Z/postprocess/README.md](/home/ka/CKM/radio-unet/runs/width32_b128_20260908T052359Z/postprocess/README.md)

For all 160 test samples in each group, the export saves prediction/ground-truth/valid-mask NPZ files, condition NPZ files, comparison and error figures (220-dpi PNG and PDF), JSON metadata, and file digests. Original test metrics are not overwritten. Export recomputes them with the same best EMA checkpoint and inference parameters for verification. Each array's final output is under `visualization_results/` in its run directory; `index.html` is the browsing entry point.

The background export initially waited for the full training queue and for GPU 5/6 to become idle. `postprocess/status.json` records its state. Validation previews are not final test results. New runs must explicitly attach their own export process at launch; this sidecar does not automatically take over unrelated experiments.

2026-09-08 parallel-deployment update: 16x16 was moved to GPU 0/1 while 8x8 ran on GPU 5/6. The export watcher tracked coordination PID 509373 and waited for the parallel queue to finish before using one idle GPU from 0/1/5/6. See the [parallel-deployment record](/home/ka/CKM/radio-unet/runs/width32_b128_20260908T052359Z/parallel/README.md).

2026-09-08 post-training validation update: At the user's request, all 160 test scenes for 8x8 and 16x16 were exported early on idle GPU 3 and GPU 0, respectively, without waiting for 32x32. Recomputed metrics exactly matched the original test records. Both archives contain 300-dpi PNG, PDF/SVG, NPZ, and per-figure QA. The exporter's default PNG resolution was raised from 220 to 300 dpi without changing the training source fingerprint. See the [visualization validation report](/home/ka/CKM/radio-unet/runs/width32_b128_20260908T052359Z/validation/VALIDATION_REPORT.md). The existing watcher will later verify and reuse these two completed exports.
