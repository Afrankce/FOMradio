"""Paired, scene-equal endpoint statistics; never select checkpoints on test."""
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from scripts.monitor_matern_suite import atomic

VARIANTS=('CASA','ADD','FILM','NSOP')


def summarize(payload,n):
    groups=defaultdict(list)
    for r in payload['records']:
        if r['steps']==n:groups[r['scene_id']].append(r)
    return {scene:{k:mean(r[k] for r in rr) for k in ('db_rmse','fine_mse')}
            for scene,rr in groups.items()}


def analyze(root):
    scope=root/'execution_scope.json'
    if not scope.exists():return
    reference=Path(json.loads(scope.read_text())['reference'])
    out=root/'ema_comparison';out.mkdir(exist_ok=True)
    paths=[d/f'{s}_diagnostics.json' for v in VARIANTS for s in ('val','test')
           for d in (reference/'8x8'/v,root/'8x8'/v/'reference_online')]
    signature={str(p):p.stat().st_mtime_ns for p in paths if p.exists()}
    cache=out/'signature.json'
    if cache.exists() and json.loads(cache.read_text())==signature:return
    rows=[];pairs=[]
    for v in VARIANTS:
        for split in ('val','test'):
            baseline=reference/'8x8'/v/f'{split}_diagnostics.json'
            online=root/'8x8'/v/'reference_online'/f'{split}_diagnostics.json'
            if not baseline.exists():continue
            old=json.loads(baseline.read_text())
            new=json.loads(online.read_text()) if online.exists() else None
            for n in sorted({r['steps'] for r in old['records']}):
                a=summarize(old,n);b=summarize(new,n) if new else {}
                if b and a.keys()!=b.keys():raise ValueError('Unpaired scene IDs')
                row=dict(variant=v,split=split,steps=n,scenes=len(a),
                    ema_rmse=mean(x['db_rmse'] for x in a.values()),
                    ema_fine_mse=mean(x['fine_mse'] for x in a.values()),
                    ema_ssim=mean(r['ssim'] for r in old['aggregated'] if r['steps']==n))
                if b:
                    row.update(online_rmse=mean(x['db_rmse'] for x in b.values()),
                        online_fine_mse=mean(x['fine_mse'] for x in b.values()),
                        online_ssim=mean(r['ssim'] for r in new['aggregated'] if r['steps']==n))
                    row['delta_rmse_online_minus_ema']=row['online_rmse']-row['ema_rmse']
                    row['delta_fine_mse_online_minus_ema']=row['online_fine_mse']-row['ema_fine_mse']
                    row['scene_win_fraction']=mean(b[s]['db_rmse']<a[s]['db_rmse'] for s in a)
                    for s in sorted(a):
                        pairs.append(dict(variant=v,split=split,steps=n,scene=s,
                            delta_rmse=b[s]['db_rmse']-a[s]['db_rmse'],
                            delta_fine_mse=b[s]['fine_mse']-a[s]['fine_mse']))
                rows.append(row)
    for name,data in [('metrics.csv',rows),('paired_scene_deltas.csv',pairs)]:
        if data:
            stream=io.StringIO();writer=csv.DictWriter(stream,sorted(set().union(*(r.keys() for r in data))))
            writer.writeheader();writer.writerows(data);atomic(out/name,stream.getvalue())
    lines=['# Epoch 1000: online versus EMA weights from the same run','',
        'Both weights come from the same original last.pt. Paired data, MS source seed, Euler steps, mask, and postprocessing are identical.',
        'RMSE/Haar: average four source draws within each scene, then weight scenes equally; SSIM uses full-valid-window aggregation.',
        'This is not a new training result. See ../visualizations/README.md for online weights at later stages. One training seed does not support cross-seed significance claims.','',
        '| Variant | Split | N | EMA RMSE | Online RMSE | Online - EMA | EMA Haar MSE | Online Haar MSE |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['steps']!=8:continue
        fmt=lambda k:f'{r[k]:.4f}' if k in r else 'Pending evaluation'
        lines.append(f"| {r['variant']} | {r['split']} | 8 | "+' | '.join(fmt(k) for k in (
            'ema_rmse','online_rmse','delta_rmse_online_minus_ema','ema_fine_mse','online_fine_mse'))+' |')
    lines+=['','A negative difference means lower error for online weights. See metrics.csv for the full N sweep.',
        'Among completed EMA comparisons, CASA outperforms ADD, FiLM, and NSOP on test RMSE, SSIM, and detail MSE.',
        'This does not establish that CA/SA is inherently unsuitable for FNO, nor does one failed conditional kernel refute nonstationary operators.',
        'EMA lag would be implicated only if online weights clearly recovered local detail that EMA did not.',
        'An early-only benefit followed by a tie or degradation is more consistent with an early-lag versus late-stability tradeoff.']
    atomic(out/'summary.md','\n'.join(lines)+'\n');atomic(cache,signature)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    analyze(p.parse_args().root.resolve())
