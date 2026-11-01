# Release notes

## `release/multiscale-uno-fm`

This focused release was prepared from the clean training branch after the three-array experiment was completed.

### Retained

- Unified training, validation, and test entry point for 6.7 GHz, 0-degree, single-beam Multiscale UNO-FM.
- Data contract, manifest builder, valid-region mask, and normalization.
- Multiscale condition encoder, CA/SA fusion, and FNO/UNO velocity network.
- AdamW, AMP, gradient accumulation, EMA, early stopping, and strict checkpoint recovery.
- Fixed hash-based noise, CFG, and two-step Euler sampling.
- Metrics, per-sample artifacts, runtime accounting, and visualization logic.
- Frozen schema, architecture diagram, reproduction instructions, and fixed-protocol results.

### Removed

- `tests/`, `pytest.ini`, and test outputs.
- Historical training/evaluation entry points and scripts with absolute server paths.
- Other experiment branches, including sparse Task 2, cross-frequency, and common8 multibeam experiments.
- Checkpoints, result directories, datasets, caches, and figures from old reports.

### Low-risk cleanup

`ModelEMA` was moved unchanged from the old monolithic `train.py` to `training/ema.py`. The two trainers changed only their import location; the update formula and state-dict behavior are unchanged. The release code therefore no longer imports the legacy ODE, data loader, or CLI solely to obtain EMA.

Apart from this dependency isolation, the model architecture, random seeds, data order, precision, training hyperparameters, loss, and sampling equations remain unchanged.
