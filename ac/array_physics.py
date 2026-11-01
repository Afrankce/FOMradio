"""Independent vectorized reproduction of the published Step6 LoS formula.

The calibration reads only beam maps and the public generation script, not
radio-map labels.  The official archive is never modified.
"""
import argparse
import importlib.util
import json
import hashlib
from pathlib import Path

import numpy as np

C0 = 3.0e8
FREQ = 6.7e9
ANGLES = (-14, 0, 14)


def offsets(n, freq=FREQ):
    # Step6 uses n/2 rather than (n-1)/2 as the reference offset.
    row, col = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    scale = .5 * C0 / freq
    return np.column_stack((np.zeros(n*n), (col.ravel()-n/2)*scale,
                            (row.ravel()-n/2)*scale))


def codebook(n, angle_deg, subarray=None):
    subarray = n if subarray is None else subarray
    if n % subarray: raise ValueError("invalid subarray")
    row, col = np.meshgrid(np.arange(subarray), np.arange(subarray), indexing="ij")
    phase = np.pi * col.ravel() * np.sin(np.deg2rad(-angle_deg))
    v = np.tile(np.exp(1j*phase), (n // subarray)**2)
    return v / np.linalg.norm(v)


def receivers():
    x = np.arange(128)*10.0-635.0
    y = np.arange(128)*10.0-635.0
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack((xx.ravel(), yy.ravel(), np.full(xx.size, 1.5)), axis=1)


def reproduce_beams(n, angle_list=ANGLES, power_dbm=0., plane=True, subarray=None, batch=128):
    """Step6 gain, phase, offsets and orientation; multiple beams per pass."""
    xyz = receivers()
    center = np.array((0., 0., 40.))
    off = offsets(n)
    weights = np.stack([codebook(n, a, subarray) for a in angle_list], axis=1)
    out = np.empty((xyz.shape[0], len(angle_list)), dtype=np.float64)
    wavelength = C0/FREQ
    for first in range(0, len(xyz), batch):
        pos = xyz[first:first+batch]
        vec = pos[:, None, :] - center[None, None, :] - off[None, :, :]
        distance = np.linalg.norm(vec, axis=-1) + 1e-30
        az = np.degrees(np.arctan2(vec[..., 1], vec[..., 0]))
        phi = 90.0 - np.degrees(np.arcsin(vec[..., 2]/distance))
        ah = -np.minimum(12*(az/65.)**2, 30.)
        av = -np.minimum(12*((phi-90.)/65.)**2, 30.)
        gain = 10**((8.-np.minimum(-(ah+av),30.))/10.)
        vcenter = pos-center
        dref = np.linalg.norm(vcenter, axis=1) + 1e-30
        if plane:
            direction = vcenter/dref[:,None]
            dphase = dref[:,None] - direction @ off.T
            damp = dref[:,None]
        else:
            dphase = damp = distance
        h = np.sqrt(gain)/damp*np.exp(2j*np.pi*dphase/wavelength)
        response = h @ weights.conj()
        p_watt = 10**((power_dbm-30)/10)*(wavelength/(4*np.pi))**2*np.abs(response)**2
        out[first:first+len(pos)] = 10*np.log10(p_watt/1e-3+1e-30)
    return out.T.reshape(len(angle_list),128,128)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official(step6):
    spec = importlib.util.spec_from_file_location('u6g_step6', step6)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--step6', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    official = load_official(args.step6)
    # Compare independent vector implementation with public scalar routine.
    samples = np.array([0, 63, 4096, 8256, 13000, 16383])
    rx = receivers()
    center = np.array([0.,0.,40.])
    assert np.allclose(rx, official.build_rx_positions(1.5))
    scalar = official.compute_beammap_los(rx[samples], center,
        official.build_tx_antenna_offsets(8,8,FREQ),
        official.generate_extended_codebook(8,8,-14,0,8,8,'rect',True),
        FREQ, True, True, 0.)
    vector = reproduce_beams(8,(-14,))[0].ravel()[samples]
    agreement = float(np.max(np.abs(scalar-vector)))
    if agreement > 1e-8: raise AssertionError(('scalar/vector mismatch',agreement))
    report = dict(official_step6_sha256=sha(args.step6), scalar_max_abs_db=agreement,
                  array_offset_origin='official Step6: row/column minus N/2',
                  data_settings_power_dbm=23, step6_default_power_dbm=0,
                  note='23 dBm metadata and Step6 0 dBm default are distinct; labels not read', cases=[])
    for n in (8,32):
        config = f'freq_6.7GHz_{n*n}TR_{8 if n==8 else 64}beams_pattern_tr38901'
        generated = reproduce_beams(n)
        for i, angle in enumerate(ANGLES):
            idx = (angle+28)//7 if n==8 else angle+32
            path=args.dataset/'beam_maps'/config/'u0'/f'beam_{idx:02d}_angle_{float(angle):.1f}_matrix.npy'
            published=np.load(path,allow_pickle=False)
            pred=generated[i]
            mask=np.isfinite(pred)&np.isfinite(published)&(published>-200)&(published<-20)
            e=np.abs(pred[mask]-published[mask])
            if e.size==0: raise ValueError(f'empty calibration mask {path}')
            row=dict(array=n, angle=angle, beam_index=idx, beam_path=str(path), beam_sha256=sha(path),
                     n_pixels=int(e.size), mae_db=float(e.mean()), p95_db=float(np.percentile(e,95)),
                     max_db=float(e.max()), signed_median_db=float(np.median(pred[mask]-published[mask])),
                     threshold_pass=bool(e.mean()<=.1 and np.percentile(e,95)<=.5))
            report['cases'].append(row)
            print(json.dumps(row), flush=True)
    report['passed']=all(x['threshold_pass'] for x in report['cases'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2), encoding='utf-8')
    print('CALIBRATION_PASS',report['passed'],flush=True)


if __name__=='__main__': main()
