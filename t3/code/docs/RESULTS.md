# Completed results

## Experimental protocol

This page covers only **6.7 GHz, 0-degree, single-beam Multiscale UNO-FM** with a 560/80/160 scene split. It does not combine results from common8 multibeam, cross-frequency, or sparse Task 2 experiments.

All three arrays use the same seed-42 scene split and training configuration, the best EMA checkpoint, CFG=1.0, fixed hash-based initial noise, and two-step Euler sampling. These test-set results come from one fixed server run; they are not means or confidence intervals over multiple random seeds.

## Test-set results

| Array | 0-degree beam ID | Best epoch | Completed epochs | dB-RMSE ↓ | dB-MAE ↓ | NMSE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 4 | 735 | 755 | 9.2108 | 5.5413 | 0.0026568 | 30.2565 | 0.9099 |
| 16x16 | 8 | 734 | 754 | 9.4604 | 5.6524 | 0.0029507 | 30.0243 | 0.9043 |
| 32x32 | 32 | 764 | 784 | 9.6818 | 5.8408 | 0.0034835 | 29.8233 | 0.8937 |

## Model size

- PyTorch parameter elements: 3,059,355, counting each complex parameter as one element.
- Independent real scalars: 3,925,659, counting each complex parameter as two real scalars.
- State channels: `(32,64,128,256,256)`.
- FNO operator width: 24.
- Modes: `(12,12,8,4,4)`.
- Nine FNO stages and nine CA/SA condition-fusion modules.

## Interpretation limits

- Results characterize only this fixed split and one training seed.
- The schema selects the physical configuration and 0-degree beam ID separately for each array.
- These numbers should not be compared directly in one primary table with multibeam, cross-frequency, or sparse-reconstruction protocols.
- Claims of statistical significance require additional independent scene splits and training seeds.
