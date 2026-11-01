# Exploratory U6G A/C test

Two seeds (42, 43), 160 held-out scenes, four paired draws, Euler NFE 8.
Positive C minus A means C has higher MAE. Intervals resample scenes, not training seeds.

| Group | A MAE dB | C MAE dB | C − A dB | 95% scene bootstrap CI |
|---|---:|---:|---:|---:|
| seen_config | 5.855 | 7.666 | +1.811 | [+1.739, +1.881] |
| array | 11.545 | 16.719 | +5.174 | [+4.940, +5.400] |
| angle_interpolation | 6.371 | 11.297 | +4.926 | [+4.645, +5.200] |
| angle_extrapolation | 6.421 | 11.534 | +5.114 | [+4.850, +5.370] |
| joint | 11.329 | 16.636 | +5.306 | [+5.065, +5.541] |

This is not a confirmatory three-seed analysis; the user chose two seeds after the original plan.
C has an additional environment network, array metadata and mean loss; differences are system-level.
SSIM uses raw dB and only fully valid 11x11 windows; missing supports are reported in summary.json.
