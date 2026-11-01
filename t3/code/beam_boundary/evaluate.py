"""Paired-source validation/test, geometry-only strata and morphology."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy import ndimage
import torch
import torch.distributed as dist
from torch.nn import functional as F
from model.matern_source import SourceSampler
from rf_baseline.metrics import gradients, haar, orientation
from .common import atomic, read, seed_for, load_data, batch, checkpoint, identity, archived_identity, verify
from .model import BoundaryUNO


def metrics(pred, truth, valid, height, spacing=5.):
    error = np.asarray(pred, np.float64) - truth; values = error[valid]
    if not values.size: raise RuntimeError('No evaluation support')
    result = dict(rmse_db=float(np.sqrt(np.mean(values**2))), mae_db=float(np.mean(abs(values))),
                  bias_db=float(values.mean()), valid_pixels=int(valid.sum()))
    gx, gy, support = gradients(error, valid, spacing)
    result['gradient_rmse_db_per_m'] = float(np.sqrt(np.mean((gx[support]**2 + gy[support]**2) / 2))) if support.any() else None
    for level, detail, safe in haar(error, valid):
        result[f'haar{level}_rmse_db'] = float(np.sqrt(np.mean(detail[:, safe]**2))) if safe.any() else None
    pa, pc, _, ps = orientation(pred, valid, spacing); ta, tc, _, ts = orientation(truth, valid, spacing)
    safe = ps & ts & (tc >= .2); reference = ts & (tc >= .2)
    difference = .5 * abs(np.angle(np.exp(2j * (pa - ta))))
    result['orientation_mae_deg'] = float(np.sum(difference[safe] * tc[safe]) / tc[safe].sum() * 180 / np.pi) if safe.any() else None
    result['orientation_coverage'] = float(safe.sum() / reference.sum()) if reference.any() else None
    building = height > 1. / 445.
    distance = ndimage.distance_transform_edt(~building, sampling=spacing) if building.any() else np.full(height.shape, np.inf)
    for name, region in [('near', distance <= 20), ('open', distance > 50)]:
        support = valid & region; result[f'{name}_pixels'] = int(support.sum())
        result[f'{name}_rmse_db'] = float(np.sqrt(np.mean(error[support]**2))) if support.any() else None
    return result


def response(pred, reference, truth, truth_reference, valid):
    p = (pred - reference)[valid].astype(np.float64); y = (truth - truth_reference)[valid].astype(np.float64)
    error = p - y; pc = p - p.mean(); yc = y - y.mean()
    return dict(delta_rmse_db=float(np.sqrt(np.mean(error**2))), delta_bias_db=float(error.mean()),
                delta_debiased_rmse_db=float(error.std()),
                response_gain=float(np.dot(pc, yc) / np.dot(yc, yc)) if np.dot(yc, yc) > 1e-12 else None,
                response_correlation=float(np.dot(pc, yc) / np.sqrt(np.dot(pc, pc) * np.dot(yc, yc)))
                if np.dot(pc, pc) * np.dot(yc, yc) > 1e-12 else None)


def summarize(rows, delta_rows):
    def mean(items, key):
        values = [r[key] for r in items if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    unseen = [r for r in delta_rows if r['angle'] not in (-14., 0., 14.)]
    result = dict(map_rmse_db=mean(rows, 'rmse_db'), unseen_delta_rmse_db=mean(unseen, 'delta_rmse_db'),
        scenes=len({r['scene'] for r in rows}), maps=len(rows),
        aggregation='Equal scene, angle and source-draw weight; mean per-map RMSE, not pooled-pixel RMSE')
    for category, angles in [('interpolation', (-7., 7.)), ('extrapolation', (-28., -21., 21.)), ('seen', (-14., 14.))]:
        group = [r for r in delta_rows if r['angle'] in angles]
        result[category] = {k: mean(group, k) for k in ('delta_rmse_db', 'delta_bias_db', 'delta_debiased_rmse_db', 'response_gain', 'response_correlation')}
    for k in ('gradient_rmse_db_per_m', 'haar1_rmse_db', 'haar2_rmse_db', 'haar3_rmse_db', 'haar4_rmse_db',
              'orientation_mae_deg', 'orientation_coverage', 'near_rmse_db', 'open_rmse_db', 'native_rmse_db'):
        result[k] = mean(rows, k)
    result['unseen_delta_by_scene'] = {scene: mean([r for r in unseen if r['scene'] == scene], 'delta_rmse_db')
                                     for scene in sorted({r['scene'] for r in unseen})}
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--arm', required=True)
    p.add_argument('--base', default='E2'); p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--checkpoint-dir', type=Path); p.add_argument('--limit', type=int); p.add_argument('--draws', type=int)
    p.add_argument('--nfe', type=int); p.add_argument('--tag'); args = p.parse_args()
    root = args.root.resolve(); cfg = read(root / 'config.json'); verify(root)
    rank, world, local = map(lambda k: int(os.environ[k]), ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    torch.cuda.set_device(local); device = torch.device('cuda', local); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    folder = args.checkpoint_dir or root / 'jobs' / args.arm; saved, commit = checkpoint(folder)
    current_identity = identity(root, args.arm, args.base)
    if saved['identity'] != current_identity:
        archive = read(root / 'PACKAGE_MANIFEST.json')
        original_identity = archived_identity(root, args.arm, args.base)
        original_checkpoint = archive['checkpoint']['sha256']
        if not (args.arm == 'T3' and saved['identity'] == original_identity
                and commit['sha256'] == original_checkpoint):
            raise RuntimeError('Evaluation identity mismatch')
    torch.manual_seed(cfg['seed'])
    from cross_array.paths import KINDS, initial_state
    cross_array = cfg.get('experiment_kind') in KINDS
    if cross_array:
        from cross_array.model import SourceObservationUNO
        from cross_array.geometry import regional_metrics
        model = SourceObservationUNO(args.arm).to(device)
    else:
        model = BoundaryUNO(args.arm, args.base, cfg.get('phase_input_only', False)).to(device)
    model.load_state_dict(saved['model']); model.eval(); del saved
    step = commit['step']; tag = args.tag or f'step_{step:06d}'
    dest = root / 'jobs' / args.arm / 'evaluation' / tag / args.split; dest.mkdir(parents=True, exist_ok=True)
    arrays, records = load_data(root, args.split)
    scenes = sorted({r['scene_id'] for r in records}, key=lambda s: seed_for(42, 'visual-and-preview-scenes', args.split, s))
    if args.limit: scenes = scenes[:args.limit]
    draws, nfe = args.draws or cfg['eval_draws'], args.nfe or cfg['eval_nfe']
    source = SourceSampler('MS', resolution=256, seed=42)
    raw_root = Path(read(root / 'input_audit.json')['raw_root']); rows = []; delta_rows = []; began = time.time()
    for counter, scene in enumerate(scenes[rank::world]):
        indices = sorted([i for i, r in enumerate(records) if r['scene_id'] == scene], key=lambda i: records[i]['steering_deg'])
        c, y, mask = batch(arrays, indices, device); metas = [dict(records[i], steering_deg=0.) for i in indices]
        angles = [records[i]['steering_deg'] for i in indices]; zero = angles.index(0.)
        native = np.stack([np.load(raw_root / records[i]['radiomap_path']).squeeze() for i in indices])
        gt = y[:, 0].cpu().numpy() * 300 - 300; valid = mask[:, 0].cpu().numpy(); height = c[0, 1].cpu().numpy()
        native_valid = (native > -300) & (native < 0); outputs = []; source_hashes = []; initial_hashes = []
        regions = np.load(root / 'data/geometry' / f'{scene}.npz') if cross_array else None
        with torch.autocast('cuda', dtype=torch.float16): embedding = model.embed_model(c)
        for draw in range(draws):
            z = source.sample(metas, device=device, namespace=f'evaluation-{args.split}-draw-{draw}')
            if not torch.equal(z[0], z[-1]): raise RuntimeError('Cross-angle source pairing failed')
            from .common import tensor_sha
            source_hashes.append(tensor_sha(z[:1])); state = initial_state(c, z, cfg)
            if not torch.equal(state[0], state[-1]): raise RuntimeError('Cross-angle initial-state pairing failed')
            initial_hashes.append(tensor_sha(state[:1]))
            for index in range(nfe):
                capture = scene == scenes[0] and draw == 0 and index == nfe // 2
                if hasattr(model, 'boundary_encoder'):
                    model.boundary_encoder.capture = model.boundary_decoder.capture = capture
                with torch.autocast('cuda', dtype=torch.float16):
                    velocity = model(image=c, x=state, step=index / nfe, embedding=embedding)
                state = state + velocity.float() / nfe
                if capture and hasattr(model, 'boundary_encoder'):
                    for name in ('encoder', 'decoder'):
                        branch = getattr(model, f'boundary_{name}')
                        np.savez_compressed(dest / f'gate_{name}.npz', **branch.arrays, angle=angles[0], scene=scene, fm_time=index / nfe)
            if not torch.isfinite(state).all(): raise RuntimeError('Nonfinite generated field')
            pred = state[:, 0].cpu().numpy() * 300 - 300; outputs.append(pred)
            native_pred = F.interpolate(state, (128, 128), mode='bilinear', align_corners=False)[:, 0].cpu().numpy() * 300 - 300
            for j, angle in enumerate(angles):
                row = dict(scene=scene, angle=angle, draw=draw, **metrics(pred[j], gt[j], valid[j], height))
                row['native_rmse_db'] = float(np.sqrt(np.mean((native_pred[j][native_valid[j]] - native[j][native_valid[j]])**2)))
                if cross_array: row.update(regional_metrics(native_pred[j], native[j], native_valid[j], regions))
                rows.append(row)
                if angle != 0:
                    delta_rows.append(dict(scene=scene, angle=angle, draw=draw,
                        **response(pred[j], pred[zero], gt[j], gt[zero], valid[j] & valid[zero])))
        output_dir = dest / 'predictions'; output_dir.mkdir(exist_ok=True)
        temp = output_dir / f'{scene}.tmp.npz'; target = output_dir / f'{scene}.npz'
        extra = dict(source_observation_db=c[0, 4].cpu().numpy()*300-300,
                     source_observation_valid=c[0, 5].cpu().numpy().astype(bool),
                     shadow_proxy=regions['shadow'], shadow_transition=regions['transition']) if cross_array else {}
        if cfg.get('experiment_kind') == 'cross_array_transport':
            extra.update(initial_state_hashes=initial_hashes, source_noise_scale=cfg['source_noise_scale'])
        np.savez_compressed(temp, prediction_db=np.stack(outputs), angles=angles, valid=valid, truth_db=gt,
                            height=height, beam_normalized=c[:, 2].cpu().numpy(), source_hashes=source_hashes, **extra)
        if regions is not None: regions.close()
        os.replace(temp, target)
        atomic(folder / f'eval_progress_rank{rank}.json', dict(arm=args.arm, phase='evaluating', split=args.split,
               step=step, scenes_done=counter + 1, scenes_total=len(scenes[rank::world]), elapsed=time.time() - began, updated=time.time()))
    atomic(dest / f'rows_rank{rank}.json', dict(maps=rows, responses=delta_rows))
    dist.barrier()
    if rank == 0:
        all_maps = []; all_deltas = []
        for r in range(world):
            data = read(dest / f'rows_rank{r}.json'); all_maps.extend(data['maps']); all_deltas.extend(data['responses'])
        summary = summarize(all_maps, all_deltas)
        if cross_array:
            from cross_array.geometry import summarize_regions
            summary.update(summarize_regions(all_maps))
        atomic(dest / 'summary.json', summary)
        atomic(dest / 'complete.json', dict(step=step, split=args.split, nfe=nfe, draws=draws, scenes=len(scenes),
            checkpoint_sha256=commit['sha256'], source_pairing='same scene/draw across all angles and arms', updated=time.time()))
        print(json.dumps(dict(event='evaluation_complete', arm=args.arm, split=args.split, step=step)), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__': main()
