"""Mask-support-safe native-grid morphology; directions are not radio rays."""
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def safe_mean(x):
    return float(np.mean(x, dtype=np.float64)) if np.size(x) else None


def gradients(field, valid, spacing=10.):
    field = np.where(valid, field, 0.).astype(np.float64)
    gx = (field[1:-1, 2:] - field[1:-1, :-2]) / (2 * spacing)
    gy = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2 * spacing)
    support = ndimage.binary_erosion(valid, structure=np.ones((3, 3)), border_value=0)[1:-1, 1:-1]
    return gx, gy, support


def haar(field, valid, levels=4):
    value = np.where(valid, field, 0.).astype(np.float64)
    support = valid.copy()
    output = []
    for level in range(1, levels + 1):
        a, b, c, d = value[::2, ::2], value[::2, 1::2], value[1::2, ::2], value[1::2, 1::2]
        support = support[::2, ::2] & support[::2, 1::2] & support[1::2, ::2] & support[1::2, 1::2]
        details = np.stack(((a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2))
        output.append((level, details, support.copy()))
        value = (a+b+c+d)/2
    return output


def orientation(field, valid, spacing=10., window=7):
    gx, gy, support = gradients(field, valid, spacing)
    xx = ndimage.uniform_filter(gx * gx, window, mode='constant')
    yy = ndimage.uniform_filter(gy * gy, window, mode='constant')
    xy = ndimage.uniform_filter(gx * gy, window, mode='constant')
    trace = xx + yy
    coherence = np.sqrt((xx-yy)**2 + 4 * xy**2) / np.maximum(trace, 1e-12)
    angle = (.5 * np.arctan2(2 * xy, xx-yy)) % np.pi
    safe = ndimage.binary_erosion(support, structure=np.ones((window, window)), border_value=0) & (trace > 1e-12)
    return angle, coherence, trace, safe


def describe(field, valid, spacing=10.):
    field = np.asarray(field, np.float64); valid = np.asarray(valid, bool)
    if not np.isfinite(field).all():
        raise ValueError('Nonfinite field cannot enter diagnostics')
    result = dict(valid_fraction=float(valid.mean()), valid_pixels=int(valid.sum()),
                  mean_db=safe_mean(field[valid]),
                  out_of_range_fraction=safe_mean(((field < -300) | (field > 0))[valid]))
    for q in (5, 25, 50, 75, 95):
        result[f'power_q{q}_db'] = float(np.percentile(field[valid], q)) if valid.any() else None
    gx, gy, support = gradients(field, valid, spacing)
    mag = np.hypot(gx, gy)[support]
    result['gradient_support_pixels'] = int(support.sum())
    result['gradient_rms_db_per_m'] = float(np.sqrt(np.mean(mag**2))) if len(mag) else None
    result['gradient_p95_db_per_m'] = float(np.percentile(mag, 95)) if len(mag) else None
    for level, detail, safe in haar(field, valid):
        result[f'haar{level}_support'] = int(safe.sum())
        result[f'haar{level}_rms_db'] = float(np.sqrt(np.mean(detail[:, safe]**2))) if safe.any() else None
        for axis, label in enumerate(('column', 'row', 'diagonal')):
            result[f'haar{level}_{label}_energy_db2'] = safe_mean(detail[axis, safe]**2)
    angle, coherence, trace, safe = orientation(field, valid, spacing)
    result['orientation_support_pixels'] = int(safe.sum())
    result['orientation_coherence_mean'] = safe_mean(coherence[safe])
    directional = safe & (coherence >= .2)
    weights = trace[directional] * coherence[directional]
    hist, _ = np.histogram(angle[directional], bins=np.linspace(0, np.pi, 19), weights=weights)
    result['orientation_hist'] = (hist / hist.sum()).tolist() if hist.sum() > 0 else None
    result['directional_pixels'] = int(directional.sum())
    result['orientation_axial_concentration'] = float(abs(np.sum(weights * np.exp(2j * angle[directional])) / weights.sum())) if weights.sum() else None
    return result


def paired(pred, truth, valid, pred_class, true_class, spacing=10.):
    # Every value metric uses TRUE support, never predicted support to hide errors.
    error = np.asarray(pred, np.float64) - truth
    out = dict(rmse_db=float(np.sqrt(np.mean(error[valid]**2))), mae_db=float(np.mean(abs(error[valid]))))
    for cls, label in enumerate(('valid', 'building', 'unlabelled')):
        union = (pred_class == cls) | (true_class == cls)
        out[f'{label}_iou'] = float(((pred_class == cls) & (true_class == cls)).sum() / union.sum()) if union.any() else None
    dx, dy, support = gradients(error, valid, spacing)
    out['gradient_rmse_db_per_m'] = float(np.sqrt(np.mean((dx[support]**2 + dy[support]**2)/2))) if support.any() else None
    for level, detail, safe in haar(error, valid):
        out[f'haar{level}_rmse_db'] = float(np.sqrt(np.mean(detail[:, safe]**2))) if safe.any() else None
    pa, pc, _, ps = orientation(pred, valid, spacing)
    ta, tc, _, ts = orientation(truth, valid, spacing)
    safe = ps & ts & (tc >= .2)
    reference = ts & (tc >= .2)
    out['orientation_comparable_pixels'] = int(safe.sum())
    out['orientation_reference_pixels'] = int(reference.sum())
    out['orientation_comparable_fraction'] = float(safe.sum()/reference.sum()) if reference.any() else None
    difference = .5 * np.abs(np.angle(np.exp(2j * (pa-ta))))
    out['orientation_mae_deg'] = float(np.sum(difference[safe] * tc[safe]) / tc[safe].sum() * 180 / np.pi) if safe.any() else None
    out['orientation_coherence_mae'] = safe_mean(np.abs(pc[ts]-tc[ts]))
    # Geometry strata are from the target's building labels, not target power.
    building = true_class == 1
    distance = ndimage.distance_transform_edt(~building, sampling=spacing) if building.any() else np.full(valid.shape, np.inf)
    for name, region in (('near_building', distance <= 20), ('open', distance > 50)):
        support = valid & region
        out[name + '_pixels'] = int(support.sum())
        out[name + '_rmse_db'] = float(np.sqrt(np.mean(error[support]**2))) if support.any() else None
    return out


SUMMARY_FEATURES = ('mean_db', 'power_q5_db', 'power_q95_db', 'valid_fraction',
                    'gradient_rms_db_per_m', 'gradient_p95_db_per_m',
                    'haar1_rms_db', 'haar2_rms_db', 'haar3_rms_db', 'haar4_rms_db',
                    'orientation_coherence_mean', 'orientation_axial_concentration',
                    'gradient_support_pixels', 'haar1_support', 'haar2_support',
                    'haar3_support', 'haar4_support', 'directional_pixels')


def distribution_metrics(real, generated):
    out = {'real_maps': len(real), 'generated_maps': len(generated), 'features': {}}
    for name in SUMMARY_FEATURES:
        a = [r[name] for r in real if r.get(name) is not None]
        b = [r[name] for r in generated if r.get(name) is not None]
        out['features'][name] = dict(real_n=len(a), generated_n=len(b),
                real_mean=safe_mean(a), generated_mean=safe_mean(b),
                wasserstein1=float(wasserstein_distance(a, b)) if a and b else None)
    a = [r['orientation_hist'] for r in real if r.get('orientation_hist') is not None]
    b = [r['orientation_hist'] for r in generated if r.get('orientation_hist') is not None]
    out['orientation_hist_valid_maps'] = dict(real=len(a), generated=len(b))
    if a and b:
        ah, bh = np.mean(a, 0), np.mean(b, 0)
        out.update(orientation_jsd_nats=float(jensenshannon(ah, bh)**2),
                   real_orientation_hist=ah.tolist(), generated_orientation_hist=bh.tolist())
    else:
        out['orientation_jsd_nats'] = None
    return out
