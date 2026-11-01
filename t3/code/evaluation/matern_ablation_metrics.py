"""Pre-registered paired-scene gate and mask-safe detail diagnostics.

Haar errors are computed AFTER identical 2x2 averaging of the 256-grid target
and prediction. This is a proxy on the interpolated benchmark, not recovered
native 128-grid observations. Every coefficient support must be wholly valid.
"""
import numpy as np
import torch
from torch.nn import functional as F


def haar_step(value, mask):
    a, b = value[..., ::2, ::2], value[..., ::2, 1::2]
    c, d = value[..., 1::2, ::2], value[..., 1::2, 1::2]
    valid = F.avg_pool2d(mask.float(), 2) == 1
    return (a + b + c + d) / 2, torch.cat(((a - b + c - d) / 2,
            (a + b - c - d) / 2, (a - b - c + d) / 2), dim=1), valid


def scene_diagnostics(prediction, target, mask):
    if prediction.shape[0] != 1:
        raise ValueError("Scene-level metrics require one scene")
    if not torch.isfinite(prediction[mask]).all():
        raise ValueError("Nonfinite prediction")
    error = (prediction.clamp(0, 1).double() - target.double()) * 300
    raw = (prediction.double() - target.double()) * 300
    selected = error[mask]
    result = dict(db_rmse=selected.square().mean().sqrt().item(), db_bias=selected.mean().item(),
                  raw_db_rmse=raw[mask].square().mean().sqrt().item(), valid_pixels=int(mask.sum()))
    p = F.avg_pool2d(prediction.clamp(0, 1).double() * 300, 2)
    y = F.avg_pool2d(target.double() * 300, 2)
    valid = F.avg_pool2d(mask.float(), 2) == 1
    fine_sum, fine_count = 0., 0
    for level in range(1, 5):
        p, pd, next_valid = haar_step(p, valid)
        y, yd, _ = haar_step(y, valid)
        support = next_valid.expand_as(pd)
        count = int(support.sum())
        squared = (pd - yd)[support].square().sum().item()
        energy = yd[support].square().sum().item()
        result[f"haar_{level}_count"] = count
        result[f"haar_{level}_mse"] = squared / count if count else None
        result[f"haar_{level}_target_energy"] = energy / count if count else None
        if level <= 2:
            fine_sum += squared
            fine_count += count
        valid = next_valid
    result["fine_mse"] = fine_sum / fine_count if fine_count else None
    result["fine_count"] = fine_count
    # Pairwise valid semivariograms, independent of centering, on 256 benchmark.
    for lag in (2, 4, 8, 16):
        semis = []
        for field in (prediction.clamp(0, 1), target):
            numerator, count = 0., 0
            for axis in (-2, -1):
                a, b = [slice(None)] * 4, [slice(None)] * 4
                a[axis], b[axis] = slice(lag, None), slice(None, -lag)
                a, b = tuple(a), tuple(b)
                support = mask[a] & mask[b]
                numerator += (((field[a] - field[b]).double() * 300)[support].square().sum().item())
                count += int(support.sum())
            semis.append(numerator / (2 * count) if count else None)
        result[f"variogram_{lag}_pred"], result[f"variogram_{lag}_target"] = semis
    return result


def scene_means(records, steps):
    grouped = {}
    for row in records:
        if row["steps"] == steps:
            grouped.setdefault(row["scene_id"], []).append(row)
    if len(grouped) != 80 or any(len(rows) != 4 or {r["realization"] for r in rows} != {0, 1, 2, 3}
                                 for rows in grouped.values()):
        raise ValueError("Gate requires all 80 validation scenes and four distinct realizations per scene")
    return {scene: {key: float(np.mean([r[key] for r in rows])) for key in ("fine_mse", "db_rmse", "db_bias")}
            for scene, rows in grouped.items()}


def validation_gate(results):
    """Never accepts test records. Thresholds fixed before any model training."""
    if set(results) != {"B2", "M1", "M2", "M3"}:
        raise ValueError("All four ablations required")
    if any(item["split"] != "val" or item["array_size"] != "8x8" for item in results.values()):
        raise ValueError("Only 8x8 validation evidence may authorize the 16x16 run")
    summaries = {name: {n: scene_means(item["records"], n) for n in (2, 8)} for name, item in results.items()}
    comparisons = []
    for parent, child in (("M1", "M2"), ("M2", "M3"), ("B2", "M3")):
        a, b = summaries[parent][2], summaries[child][2]
        if set(a) != set(b):
            raise ValueError("Scene sets differ")
        scenes = sorted(a)
        before = np.array([a[s]["fine_mse"] for s in scenes])
        after = np.array([b[s]["fine_mse"] for s in scenes])
        if not np.isfinite(before).all() or not np.isfinite(after).all() or before.mean() <= 0:
            raise ValueError("Invalid fine-detail gate evidence")
        difference = before - after
        indices = np.random.default_rng(20260909).integers(0, len(scenes), (10000, len(scenes)))
        lo, hi = np.quantile(difference[indices].mean(axis=1), [.025, .975])
        fraction = difference.mean() / before.mean()
        rmse_delta = float(np.mean([b[s]["db_rmse"] - a[s]["db_rmse"] for s in scenes]))
        bias_delta = float(np.mean([abs(b[s]["db_bias"]) - abs(a[s]["db_bias"]) for s in scenes]))
        a8, b8 = summaries[parent][8], summaries[child][8]
        if set(a8) != set(a) or set(b8) != set(a):
            raise ValueError("Step sweep scene sets differ")
        rmse_delta8 = float(np.mean([b8[s]["db_rmse"] - a8[s]["db_rmse"] for s in scenes]))
        passed = fraction >= .02 and lo > 0 and rmse_delta <= .1 and rmse_delta8 <= .1 and bias_delta <= .1
        comparisons.append(dict(parent=parent, child=child, relative_fine_mse_reduction=float(fraction),
                                paired_scene_fine_mse_delta_ci95=[float(lo), float(hi)],
                                scene_rmse_delta_db=rmse_delta, scene_abs_bias_delta_db=bias_delta,
                                n8_scene_rmse_delta_db=rmse_delta8, passed=bool(passed)))
    return dict(status="passed" if all(r["passed"] for r in comparisons) else "not_supported",
                launch_16x16_m3=all(r["passed"] for r in comparisons), comparisons=comparisons,
                independent_training_seeds=1, scope="exploratory_component_evidence_not_mechanism_proof")
