import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from beam_boundary.common import atomic, read, sha, verify, LABELS
from .geometry import regions_from_height

SOURCE = 'freq_6.7GHz_64TR_8beams_pattern_tr38901'
TARGET = 'freq_6.7GHz_1024TR_64beams_pattern_tr38901'


def normalize_map(value):
    value = np.asarray(value, np.float32)
    assert value.shape == (128, 128) and np.isfinite(value).all()
    valid = (value > -300) & (value < 0)
    assert (valid | (value == 1000) | (value == -300)).all()
    field = torch.from_numpy(np.where(valid, (value+300)/300., 0.))[None, None]
    resized = F.interpolate(field, (256, 256), mode='bilinear', align_corners=False)[0]
    mask = F.interpolate(torch.from_numpy(valid.astype(np.float32))[None, None], (256, 256), mode='nearest')[0].bool()
    return torch.where(mask, resized, 0.), mask


def prepare(root, parent, project):
    if root.exists(): raise RuntimeError('Refusing to overwrite a run')
    verify(parent)
    root.mkdir(parents=True); folder = root / 'data'; folder.mkdir(); (folder / 'geometry').mkdir()
    raw = Path(read(parent / 'input_audit.json')['raw_root'])
    source_config = raw / 'configs' / f'{SOURCE}_beam04_beam_settings.txt'
    target_config = raw / 'configs' / f'{TARGET}_beam32_beam_settings.txt'
    source_text, target_text = source_config.read_text(), target_config.read_text()
    for literal in ('frequency_hz: 6700000000.0', 'position: [0, 0, 40]', 'height_m: 1.5',
                    'power_dbm: 23', 'polarization: V', 'spacing: 0.5λ'):
        assert literal in source_text and literal in target_text, literal
    assert 'num_rows: 8' in source_text and 'num_cols: 8' in source_text
    assert 'num_rows: 32' in target_text and 'num_cols: 32' in target_text
    assert 'start_angle_deg: -32.0' in target_text and 'beam_spacing_deg: 1.0' in target_text
    torch.set_num_threads(2); hashes = {}; split_info = {}; regions_scenes = set(); source_fixed = {}
    def load(rel):
        path = raw / rel
        if rel not in hashes: hashes[rel] = sha(path)
        return np.load(path, allow_pickle=False)
    resize = lambda x: F.interpolate(torch.as_tensor(x)[None, None].float(), (256, 256), mode='bilinear', align_corners=False)[0]
    source_beam_rel = f'beam_maps/{SOURCE}/u0/beam_04_angle_0.0_matrix.npy'
    source_beam = resize((np.clip(load(source_beam_rel), -300, 0)+300)/300)
    tx = torch.zeros(1, 256, 256); tx[:, 127, 127] = 1

    @lru_cache(maxsize=4)
    def scene_inputs(scene):
        hrel = f'height_maps/{scene}/{scene}_height_matrix.npy'
        srel = f'radiomaps/{SOURCE}_beam04/{scene}_labeled_radiomap.npy'
        h = load(hrel); assert h.shape == (256,256) and np.isfinite(h).all() and h.min() >= 0 and h.max() <= 445
        source, source_valid = normalize_map(load(srel))
        source_fixed[scene] = dict(path=srel, sha256=hashes[srel], angle=0., beam=4)
        return resize(h/445.), source, source_valid

    target_beams = {}
    angles = [-28., -21., -14., -7., 0., 7., 14., 21.]
    for angle in angles:
        beam = int(angle+32)
        rel = f'beam_maps/{TARGET}/u0/beam_{beam:02d}_angle_{angle:.1f}_matrix.npy'
        value = load(rel); assert np.isfinite(value).all()
        target_beams[angle] = (rel, resize((np.clip(value, -300, 0)+300)/300))
        cfgrel = f'configs/{TARGET}_beam{beam:02d}_beam_settings.txt'
        text = (raw/cfgrel).read_text()
        for literal in ('num_rows: 32', 'position: [0, 0, 40]', 'height_m: 1.5', 'power_dbm: 23'):
            assert literal in text
        hashes[cfgrel] = sha(raw/cfgrel)
    hashes[str(source_config.relative_to(raw))] = sha(source_config)
    for split in ('train', 'val', 'test'):
        originals = read(parent / 'data' / f'{split}_records.json'); records = []
        arrays = {key: np.lib.format.open_memmap(folder/f'{split}_{key}.npy', mode='w+', dtype=dtype,
            shape=(len(originals), channels, 256, 256)) for key,dtype,channels in
            [('condition', np.float32,6), ('target',np.float32,1), ('valid',bool,1)]}
        for index, previous in enumerate(originals):
            scene, angle = previous['scene_id'], float(previous['steering_deg'])
            assert previous['array_rows'] == previous['array_cols'] == 8 and previous['frequency_hz'] == 6700000000
            height, source, source_valid = scene_inputs(scene)
            assert hashes[previous['height_path']] == previous['height_sha256']
            beam = int(angle+32); rel = f'radiomaps/{TARGET}_beam{beam:02d}/{scene}_labeled_radiomap.npy'
            target, valid = normalize_map(load(rel)); brel, target_beam = target_beams[angle]
            arrays['condition'][index] = torch.cat((tx, height, target_beam, source_beam, source, source_valid.float())).numpy()
            arrays['target'][index] = target.numpy(); arrays['valid'][index] = valid.numpy()
            record = dict(previous, sample_key=f'{scene}|32x32|freq6700000000|angle{angle}|beam{beam:02d}',
                array_name='32x32', array_rows=32, array_cols=32, config_id=TARGET, beam_id=beam,
                radiomap_path=rel, beam_map_path=brel, sha256=hashes[rel],
                source_radiomap_path=source_fixed[scene]['path'], source_sha256=source_fixed[scene]['sha256'],
                source_steering_deg=0., source_array='8x8', source_beam_map_path=source_beam_rel)
            record.pop('common_native_valid', None); records.append(record)
            if index % 200 == 0: print(json.dumps(dict(stage='cache', split=split, index=index, total=len(originals))), flush=True)
        for arr in arrays.values(): arr.flush()
        atomic(folder/f'{split}_records.json', records)
        scenes = sorted({r['scene_id'] for r in records}); split_info[split] = dict(scenes=len(scenes), records=len(records), scene_ids=scenes)
        assert len({(r['scene_id'],r['steering_deg']) for r in records}) == len(records)
        for scene in scenes:
            assert len({r['source_sha256'] for r in records if r['scene_id']==scene}) == 1
        if split != 'train': regions_scenes.update(scenes)
    sets = [set(split_info[s]['scene_ids']) for s in ('train','val','test')]
    assert list(map(len,sets)) == [560,80,160] and not any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2)))
    def geometry(scene):
        hp = raw / f'height_maps/{scene}/{scene}_height_matrix.npy'
        cp = hp.with_name(f'{scene}_height_matrix_coords.npz')
        with np.load(cp) as coords: regions = regions_from_height(np.load(hp), coords)
        np.savez_compressed(folder/'geometry'/f'{scene}.npz', **regions)
        return str(cp.relative_to(raw)), sha(cp), scene, {k:int(v.sum()) for k,v in regions.items()}
    region_coverage = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (rel,digest,scene,coverage) in enumerate(pool.map(geometry, sorted(regions_scenes))):
            hashes[rel]=digest; region_coverage[scene]=coverage
            if index % 40 == 0: print(json.dumps(dict(stage='geometry', completed=index+1,total=len(regions_scenes))),flush=True)
    cfg = read(parent/'config.json')
    cfg.update(schema=3, experiment_kind='cross_array_observation', arms=['T0','T2'], phase_arms=[], display_arms=['T0','T2'],
        labels=LABELS, parent_run=str(parent), phase_input_only=False, new_training_arms=['T0','T2'], reference_arms=[],
        allowed_gpu_ids=[4,5,6,7], phase_base='E0', source_array='8x8', target_array='32x32',
        source_observation_angle=0., condition_channels=['Tx','Height','Beam32','Beam8_0deg','Y8_0deg','Y8_valid'],
        comparison_pairs=[['T2','T0']], progression_gate=dict(unseen_delta_improvement=.01, native_near_improvement=.03,
        shadow_transition_gradient_improvement=.03, map_guardrail=1.01, min_region_scenes=20), created=time.time())
    for key in ('phase_min_delta_improvement','phase_candidates','phase_selection','validation_guardrail'):
        cfg.pop(key,None)
    atomic(root/'config.json',cfg)
    atomic(root/'input_audit.json',dict(passed=True,splits=split_info,raw_root=str(raw),raw_sha256=hashes,
        source_fixed_by_scene=source_fixed, source_is_real_observation=True, source_angle_invariant=True,
        region_coverage=region_coverage, geometry='2.5D height-derived obstruction proxy; no ray ground truth',
        cache_sha256={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
    shutil.copytree(parent/'code',root/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('beam_boundary','cross_array'):
        shutil.copytree(project/name,root/'code'/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(project/'tests/test_cross_array.py',root/'code/test_cross_array.py')
    shutil.copy2(project/'docs/CROSS_ARRAY_OBSERVATION.md',root/'PROTOCOL.md')
    atomic(root/'snapshot.json',dict(code_sha256={str(p.relative_to(root/'code')):sha(p)
        for p in sorted((root/'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
        parent_snapshot_sha256=sha(parent/'snapshot.json'),protocol_sha256=sha(root/'PROTOCOL.md')))
    for arm in ('T0','T2'): atomic(root/'jobs'/arm/'job.json',dict(arm=arm,phase='queued',updated=time.time()))
    atomic(root/'suite.json',dict(status='prepared',updated=time.time()))
    print(root,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    prepare(a.root.resolve(),a.parent.resolve(),a.project.resolve())
