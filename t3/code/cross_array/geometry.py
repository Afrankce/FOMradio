"""Geometry-only 2.5-D obstruction proxies, never reflection ground truth."""
import numpy as np
from scipy import ndimage
from rf_baseline.metrics import gradients

REGIONAL_KEYS = ('native_near_rmse_db', 'shadow_proxy_rmse_db', 'shadow_transition_rmse_db',
                 'shadow_transition_gradient_db_per_m')


def regions_from_height(height, coordinate, tx=(0., 0., 40.), rx_height=1.5):
    height = np.asarray(height, np.float64)
    assert height.shape == (256, 256)
    assert float(coordinate['resolution']) == 5.
    assert all(abs(float(coordinate[k])-v) < 1e-6 for k,v in
               [('min_x', -637.5), ('max_x', 637.5), ('min_y', -637.5), ('max_y', 637.5)])
    # Native 128 cell centres coincide with pairs of the 256 height cells.
    yy, xx = np.meshgrid(np.arange(128)*10.-635., np.arange(128)*10.-635., indexing='ij')
    flatx, flaty = xx.ravel(), yy.ravel(); blocked = np.zeros(flatx.shape, bool)
    for start in range(0, flatx.size, 512):
        x, y = flatx[start:start+512], flaty[start:start+512]
        count = max(3, int(np.ceil(np.hypot(x-tx[0], y-tx[1]).max()/2.5))+1)
        t = np.linspace(0, 1, count)[1:-1, None]
        columns = (tx[0]+t*(x-tx[0])+637.5)/5.
        rows = (tx[1]+t*(y-tx[1])+637.5)/5.
        terrain = ndimage.map_coordinates(height, [rows, columns], order=1, mode='nearest')
        line_z = tx[2]+t*(rx_height-tx[2])
        blocked[start:start+len(x)] = (terrain > line_z + 1e-9).any(0)
    building = height.reshape(128, 2, 128, 2).max((1, 3)) > rx_height
    free = ~building; shadow = blocked.reshape(128, 128) & free
    edge = np.zeros_like(shadow)
    horizontal = (shadow[:, 1:] != shadow[:, :-1]) & free[:, 1:] & free[:, :-1]
    vertical = (shadow[1:] != shadow[:-1]) & free[1:] & free[:-1]
    edge[:, 1:] |= horizontal; edge[:, :-1] |= horizontal
    edge[1:] |= vertical; edge[:-1] |= vertical
    transition = (ndimage.distance_transform_edt(~edge, sampling=10.) <= 20.) & free if edge.any() else edge
    near = (ndimage.distance_transform_edt(free, sampling=10.) <= 20.) & free if building.any() else np.zeros_like(free)
    return dict(shadow=shadow, transition=transition, near=near, free=free, boundary=edge)


def regional_metrics(pred, truth, valid, regions):
    error = pred.astype(np.float64)-truth; result = {}
    for key, maskname in [('native_near_rmse_db', 'near'), ('shadow_proxy_rmse_db', 'shadow'), ('shadow_transition_rmse_db', 'transition')]:
        support = valid & regions[maskname]
        result[key] = float(np.sqrt(np.mean(error[support]**2))) if support.any() else None
        result[key+'_pixels'] = int(support.sum())
    gx, gy, safe = gradients(error, valid, spacing=10.)
    safe &= regions['transition'][1:-1, 1:-1]
    result['shadow_transition_gradient_db_per_m'] = float(np.sqrt(np.mean((gx[safe]**2+gy[safe]**2)/2))) if safe.any() else None
    result['shadow_transition_gradient_pixels'] = int(safe.sum())
    return result


def summarize_regions(rows):
    result = {}; scenes = sorted({r['scene'] for r in rows})
    for key in REGIONAL_KEYS:
        by_scene = {s: float(np.mean([r[key] for r in rows if r['scene']==s and r[key] is not None]))
                    for s in scenes if any(r['scene']==s and r[key] is not None for r in rows)}
        result[key] = float(np.mean(list(by_scene.values()))) if by_scene else None
        result[key+'_by_scene'] = by_scene; result[key+'_scene_count'] = len(by_scene)
    return result
