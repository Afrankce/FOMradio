import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

CORE = ('E0', 'E1', 'E2', 'E3', 'E4', 'E5')
PHASE = ('P-A', 'P-AP')
LABELS = dict(E0='NoDrop AP backbone', E1='Spatial-mean source gate',
              E2='Beam-boundary Fourier residual', E3='Geometry-only source gate',
              E4='Parameter-matched local residual', E5='No amplitude lift',
              **{'P-A': 'Extra Beam magnitude gate', 'P-AP': 'Extra Beam magnitude + relative-phase gate'})
LABELS.update(T0='32x32 target FM; no source observation', T2='32x32 target FM + real 8x8 observation')
LABELS.update(T3='Same T2 velocity network; source = real Y8 + MS noise')


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_for(*args):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, args)).encode()).digest()[:8], 'big') % (2**63 - 1)


def load_data(root, split):
    folder = Path(root) / 'data'
    arrays = {key: np.load(folder / f'{split}_{key}.npy', mmap_mode='r')
              for key in ('condition', 'target', 'valid')}
    return arrays, read(folder / f'{split}_records.json')


def batch(arrays, indices, device):
    return tuple(torch.from_numpy(np.array(arrays[k][indices], copy=True)).to(device)
                 for k in ('condition', 'target', 'valid'))


def identity(root, arm, base):
    return dict(config_sha256=sha(Path(root) / 'config.json'),
                snapshot_sha256=sha(Path(root) / 'snapshot.json'), arm=arm, phase_base=base)


def archived_identity(root, arm, base):
    """Identity recorded by the unedited archive, for checkpoint evaluation only."""
    root = Path(root)
    return dict(config_sha256=sha(root / 'config.json'),
                snapshot_sha256=sha(root / 'provenance' / 'ORIGINAL_SNAPSHOT.json'),
                arm=arm, phase_base=base)


def checkpoint(folder):
    folder = Path(folder); record = read(folder / 'committed.json')
    path = folder / record['checkpoint']
    if sha(path) != record['sha256']:
        raise RuntimeError(f'Checkpoint digest mismatch: {path}')
    return torch.load(path, map_location='cpu', weights_only=False), record


def verify(root):
    root = Path(root)
    for rel, expected in read(root / 'snapshot.json')['code_sha256'].items():
        if sha(root / 'code' / rel) != expected:
            raise RuntimeError(f'Frozen source changed: {rel}')


def phase_choice(summaries):
    """Validation-only, point-estimate screening; not a significance claim."""
    baseline = summaries['E0']
    eligible = [arm for arm in ('E2', 'E3', 'E5')
                if summaries[arm]['unseen_delta_rmse_db'] <= .99 * baseline['unseen_delta_rmse_db']
                and summaries[arm]['map_rmse_db'] <= 1.01 * baseline['map_rmse_db']]
    return min(eligible, key=lambda arm: (summaries[arm]['unseen_delta_rmse_db'],
                                         summaries[arm]['map_rmse_db'], arm)) if eligible else None
