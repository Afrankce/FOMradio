"""Create a new immutable-code experiment, input cache and preregistration."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import CORE, PHASE, LABELS, atomic, read, sha, state_sha
from .model import BoundaryUNO


def architecture_audit():
    torch.set_num_threads(2); torch.manual_seed(42)
    reference = BoundaryUNO('E0').state_dict(); result = {}
    for arm in CORE + PHASE:
        torch.manual_seed(42); model = BoundaryUNO(arm)
        state = model.state_dict()
        if any(not torch.equal(value, state[key]) for key, value in reference.items()):
            raise RuntimeError(f'Common backbone initialization mismatch: {arm}')
        real = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
        result[arm] = dict(real_parameters=real, added_real_parameters=real - 5435031,
                           total_state_sha256=state_sha(state), common_backbone_sha256=state_sha(reference))
    if result['E2']['real_parameters'] != result['E4']['real_parameters']:
        raise RuntimeError('Local control parameter count not matched')
    return dict(passed=True, arms=result, common_backbone_sha256=state_sha(reference))


def prepare(root, project):
    if root.exists(): raise RuntimeError('Refusing to overwrite an experiment directory')
    root.mkdir(parents=True)
    baseline = project / 'runs/ap_nodrop_8x8_20260911T104000Z/code'
    # Use the original frozen NoDrop implementation, not unrelated current edits.
    shutil.copytree(baseline, root / 'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(project / 'beam_boundary', root / 'code/beam_boundary', ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'code/rf_baseline').mkdir(exist_ok=True)
    shutil.copy2(project / 'rf_baseline/metrics.py', root / 'code/rf_baseline/metrics.py')
    (root / 'code/rf_baseline/__init__.py').touch()
    shutil.copy2(project / 'tests/test_beam_boundary.py', root / 'code/test_beam_boundary.py')
    shutil.copy2(project / 'docs/BEAM_BOUNDARY_PROTOCOL.md', root / 'PROTOCOL.md')
    atomic(root / 'config.json', dict(schema=1, seed=42, arms=list(CORE), phase_arms=list(PHASE),
        labels=LABELS, world_size=2, global_batch=128, microbatch=16, total_steps=5000,
        milestones=[1000, 2500, 5000], checkpoint_every=250, log_every=25,
        learning_rate=.001, weight_decay=1e-5, warmup_steps=500, precision='float16_ComplexGradScaler',
        gradient_sync='fixed_order_real_view_fp64_sum', source='MS', source_length_pixels=[2, 8, 32],
        height_max=445., resolution=256, native_resolution=128, native_spacing_m=10.,
        train_angles=[-14., 0., 14.], eval_angles=[-28., -21., -14., -7., 0., 7., 14., 21.],
        eval_nfe=8, eval_draws=4, preview_scenes=16, preview_draws=1, validation_guardrail=1.01,
        phase_min_delta_improvement=.01, phase_candidates=['E2', 'E3', 'E5'],
        phase_selection='validation_unseen_delta_RMSE_then_map_RMSE_then_arm',
        amplitude_reference_db=-80., amplitude_reference_policy='fixed_numeric_reference_not_fitted_to_test',
        ema=False, loss='FM + 0.1*(Haar1+Haar2)/2 after 2x average pooling',
        created=time.time(), physical_claim='spatial power-feature phase, not electromagnetic carrier phase'))
    raw = Path('/home/ka/CKM/data/raw/49ca1dcebe2caa2b2112e6c862132243a992b00a/Dataset')
    manifests = project / 'runs/nonstationary_kernel_20260916_v1/manifests'
    folder = root / 'data'; folder.mkdir()
    tx = torch.zeros(1, 256, 256); tx[:, 127, 127] = 1
    hashes = {}; splits = {}; meta = {}
    torch.set_num_threads(2)
    for split in ('train', 'val', 'test'):
        records = [json.loads(line) for line in (manifests / f'paired_{split}.jsonl').read_text().splitlines()]
        if split == 'train': records = [r for r in records if r['steering_deg'] in (-14., 0., 14.)]
        scenes = sorted({r['scene_id'] for r in records}); splits[split] = set(scenes)
        atomic(folder / f'{split}_records.json', records)
        arrays = {key: np.lib.format.open_memmap(folder / f'{split}_{key}.npy', mode='w+', dtype=dtype,
                  shape=(len(records), channels, 256, 256))
                  for key, dtype, channels in [('condition', np.float32, 3), ('target', np.float32, 1), ('valid', bool, 1)]}
        for i, record in enumerate(records):
            values = {}
            for key in ('height', 'beam_map', 'radiomap'):
                rel = record[f'{key}_path']; path = raw / rel
                if rel not in hashes: hashes[rel] = sha(path)
                expected = record.get('height_sha256' if key == 'height' else 'sha256' if key == 'radiomap' else '_none')
                if expected and hashes[rel] != expected: raise RuntimeError(f'Input hash changed: {rel}')
                values[key] = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32)).squeeze()
                if values[key].ndim != 2 or not torch.isfinite(values[key]).all(): raise RuntimeError('Bad raw input')
            target = values['radiomap']; valid = (target > -300) & (target < 0)
            if not (valid | (target == 1000) | (target == -300)).all(): raise RuntimeError('Unknown target sentinel')
            if (values['height'] < 0).any() or (values['height'] > 445).any(): raise RuntimeError('Height normalization mismatch')
            resize = lambda v: F.interpolate(v[None, None], (256, 256), mode='bilinear', align_corners=False)[0]
            height = resize(values['height'] / 445.)
            beam = resize((values['beam_map'].clamp(-300, 0) + 300) / 300)
            target = resize(torch.where(valid, (target + 300) / 300, 0.))
            valid = F.interpolate(valid[None, None].float(), (256, 256), mode='nearest')[0].bool()
            target = torch.where(valid, target, 0.)
            arrays['condition'][i] = torch.cat((tx, height, beam)).numpy()
            arrays['target'][i] = target.numpy(); arrays['valid'][i] = valid.numpy()
            if i % 200 == 0: print(json.dumps(dict(event='cache', split=split, index=i, count=len(records))), flush=True)
        for value in arrays.values(): value.flush()
        meta[split] = dict(scenes=len(scenes), records=len(records), scene_ids=scenes)
    if [len(splits[s]) for s in ('train', 'val', 'test')] != [560, 80, 160]: raise RuntimeError('Wrong scene counts')
    if any(splits[a] & splits[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]): raise RuntimeError('Scene leakage')
    atomic(root / 'input_audit.json', dict(passed=True, splits=meta, raw_root=str(raw), raw_sha256=hashes,
         cache_sha256={p.name: sha(p) for p in folder.glob('*')}, sentinels_removed_before_fft=True))
    for arm in CORE + PHASE:
        atomic(root / 'jobs' / arm / 'job.json', dict(arm=arm, phase='queued' if arm in CORE else 'conditional_wait', updated=time.time()))
    # Run architecture checks against the actual frozen code.
    command = [sys.executable, '-c', 'from beam_boundary.prepare import architecture_audit; import json; print(json.dumps(architecture_audit()))']
    output = subprocess.run(command, cwd=root / 'code', capture_output=True, text=True, check=True)
    atomic(root / 'architecture_audit.json', json.loads(output.stdout.splitlines()[-1]))
    atomic(root / 'snapshot.json', dict(code_sha256={str(p.relative_to(root / 'code')): sha(p)
           for p in sorted((root / 'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
           baseline=str(baseline), protocol_sha256=sha(root / 'PROTOCOL.md')))
    atomic(root / 'suite.json', dict(status='prepared', updated=time.time()))
    print(str(root), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args(); prepare(a.root.resolve(), a.project.resolve())
