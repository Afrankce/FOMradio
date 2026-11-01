"""Validate complete A/C test artifacts and perform scene-paired bootstrap."""
from collections import defaultdict
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
GROUPS=('seen_config','array','angle_interpolation','angle_extrapolation','joint')
METRICS=('mae_db','rmse_db','psnr_db','ssim','boundary_mae_db','outside_range_fraction')


def scene_means(rows,metric):
    grouped=defaultdict(list)
    for row in rows:
        value=row.get(metric)
        if value is not None:
            if not np.isfinite(value):raise ValueError(f'nonfinite {metric}')
            grouped[row['scene']].append(float(value))
    return {scene:float(np.mean(values)) for scene,values in grouped.items() if values}


def paired_bootstrap(a,c,repetitions=10000,seed=20261101):
    if set(a)!=set(c) or not a:raise ValueError('seed mismatch')
    seeds=sorted(a);scenes=set(a[seeds[0]])
    if not scenes or any(set(a[s])!=scenes or set(c[s])!=scenes for s in seeds):raise ValueError('scene mismatch')
    scenes=sorted(scenes,key=lambda x:int(x[1:]))
    per_seed={s:float(np.mean([c[s][scene]-a[s][scene] for scene in scenes])) for s in seeds}
    delta=np.array([np.mean([c[s][scene]-a[s][scene] for s in seeds]) for scene in scenes],dtype=np.float64)
    rng=np.random.default_rng(seed)
    samples=delta[rng.integers(0,len(delta),size=(repetitions,len(delta)))].mean(axis=1)
    return {'mean_delta_c_minus_a':float(delta.mean()),
            'ci_95':[float(x) for x in np.quantile(samples,[.025,.975])],
            'one_sided_upper_95':float(np.quantile(samples,.95)),
            'per_seed_delta_c_minus_a':{str(k):v for k,v in per_seed.items()},
            'scenes':len(scenes),'repetitions':repetitions,'bootstrap_seed':seed,
            'scope':'scene variability only; training-seed uncertainty is not fully captured'}


def stable_hash(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@lru_cache(maxsize=256)
def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def metric_summary(rows,metric,all_scenes):
    scene=scene_means(rows,metric)
    return {'mean':float(np.mean(list(scene.values()))) if scene else None,
            'missing_images':sum(r.get(metric) is None for r in rows),
            'missing_scenes':len(all_scenes-set(scene)),'scenes_used':len(scene),
            'infinite_images':sum(bool(r.get('psnr_infinite')) for r in rows) if metric=='psnr_db' else 0}


def main():
    spec=json.loads((ROOT/'test_run_spec.json').read_text())
    output=ROOT/spec['test_output_root'];data={};identities={};support_reference={}
    for group in GROUPS:
        records=json.loads((ROOT/'manifests'/f'{group}.json').read_text())
        expected={(r['query']['scene'],r['query']['angle'],r['query']['array']) for r in records}
        if len(expected)!=len(records) or len(records)!=spec['groups_expected_queries'][group]:raise RuntimeError('manifest mismatch')
        for arm in ('A','C'):
            for seed in (42,43):
                folder=output/f'{arm}_seed{seed}'
                progress=json.loads((folder/'progress.json').read_text())
                expected_predictions=4*sum(spec['groups_expected_queries'][g] for g in GROUPS)
                if progress.get('phase')!='complete' or progress.get('completed_predictions')!=expected_predictions:
                    raise RuntimeError(f'incomplete worker {arm}/{seed}')
                ident=json.loads((folder/'identity.json').read_text());ident_hash=stable_hash(ident)
                identities[f'{arm}_{seed}']=ident
                run=ROOT/'runs'/f'{arm}_seed{seed}'
                best=json.loads((run/'best.json').read_text())
                if ident['checkpoint_sha256']!=best['checkpoint_sha256']:raise RuntimeError('best identity mismatch')
                if file_hash(run/'best.pt')!=ident['checkpoint_sha256']:raise RuntimeError('best checkpoint changed')
                if 'files_sha256' not in ident:raise RuntimeError('source identity absent')
                for name,expected_sha in ident['files_sha256'].items():
                    if file_hash(ROOT/name)!=expected_sha:raise RuntimeError(f'source changed: {name}')
                rows=[]
                for draw in range(4):
                    payload=json.loads((folder/f'{group}_draw{draw}.json').read_text())
                    if (payload['arm'],payload['seed'],payload['group'],payload['draw'],payload['identity_sha256']) != (arm,seed,group,draw,ident_hash):
                        raise RuntimeError('result identity mismatch')
                    if payload['checkpoint_step']!=best['step']:raise RuntimeError('result checkpoint step mismatch')
                    actual={(r['scene'],r['angle'],r['array']) for r in payload['rows']}
                    if actual!=expected or len(actual)!=len(payload['rows']):raise RuntimeError('result query mismatch')
                    for row in payload['rows']:
                        key=(group,row['scene'],row['angle'],row['array'])
                        support=(row['valid_pixels'],row['valid_ssim_windows'],row['boundary_valid_pixels'])
                        if key in support_reference and support_reference[key]!=support:raise RuntimeError('target support differs')
                        support_reference[key]=support
                    rows.extend(payload['rows'])
                data[(group,arm,seed)]=rows
    report={'purpose':'exploratory held-out test','seeds':[42,43],
            'seed_deviation':spec['seed_deviation'],'groups':{},'identities':identities}
    csv_rows=[];scene_records=[]
    for group in GROUPS:
        entry={'models':{},'by_angle':{}}
        for arm in ('A','C'):
            entry['models'][arm]={}
            for seed in (42,43):
                rows=data[(group,arm,seed)];scenes={r['scene'] for r in rows}
                if len(scenes)!=160:raise RuntimeError('scene count mismatch')
                metrics=METRICS+(('mu_mae_db',) if arm=='C' else ())
                scores={m:metric_summary(rows,m,scenes) for m in metrics}
                entry['models'][arm][str(seed)]=scores
                flat={'group':group,'arm':arm,'seed':seed}
                for k,v in scores.items():flat[k]=v['mean']
                csv_rows.append(flat)
                angles=sorted({r['angle'] for r in rows})
                entry['by_angle'].setdefault(arm,{})[str(seed)]={str(angle):
                    metric_summary([r for r in rows if r['angle']==angle],'mae_db',scenes)['mean'] for angle in angles}
        a={s:scene_means(data[(group,'A',s)],'mae_db') for s in (42,43)}
        c={s:scene_means(data[(group,'C',s)],'mae_db') for s in (42,43)}
        entry['mae_paired_bootstrap']=paired_bootstrap(a,c)
        for scene in sorted(a[42],key=lambda x:int(x[1:])):
            for seed in (42,43):scene_records.append({'group':group,'scene':scene,'seed':seed,
                                                       'A_mae_db':a[seed][scene],
                                                       'C_mae_db':c[seed][scene],
                                                       'C_minus_A_db':c[seed][scene]-a[seed][scene]})
        report['groups'][group]=entry
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    fields=('group','arm','seed',*METRICS,'mu_mae_db')
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(csv_rows)
    with (output/'paired_scene_mae.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=('group','scene','seed','A_mae_db','C_mae_db','C_minus_A_db'))
        w.writeheader();w.writerows(scene_records)
    lines=['# Exploratory U6G A/C test','',
           'Two seeds (42, 43), 160 held-out scenes, four paired draws, Euler NFE 8.',
           'Positive C minus A means C has higher MAE. Intervals resample scenes, not training seeds.','',
           '| Group | A MAE dB | C MAE dB | C − A dB | 95% scene bootstrap CI |',
           '|---|---:|---:|---:|---:|']
    for group in GROUPS:
        x=report['groups'][group]; b=x['mae_paired_bootstrap']
        aa=np.mean([x['models']['A'][str(s)]['mae_db']['mean'] for s in (42,43)])
        cc=np.mean([x['models']['C'][str(s)]['mae_db']['mean'] for s in (42,43)])
        lines.append(f"| {group} | {aa:.3f} | {cc:.3f} | {b['mean_delta_c_minus_a']:+.3f} | [{b['ci_95'][0]:+.3f}, {b['ci_95'][1]:+.3f}] |")
    lines+=['','This is not a confirmatory three-seed analysis; the user chose two seeds after the original plan.',
            'C has an additional environment network, array metadata and mean loss; differences are system-level.',
            'SSIM uses raw dB and only fully valid 11x11 windows; missing supports are reported in summary.json.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print((output/'report.md').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':main()
