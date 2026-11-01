"""Final no-EMA evidence: exact-replay audit, paired effects and spatial limits."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import visualize_source_suite as p
from scripts.no_ema_test_maps import maps
from scripts.no_ema_regional import regional_analysis
from scripts.visualize_conditional_operator import COLORS,VARIANTS
import matplotlib as mpl
mpl.rcParams.update({'font.family':'sans-serif','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})


def write_csv(path,rows):
    if not rows:return
    import io
    s=io.StringIO();w=csv.DictWriter(s,sorted(set().union(*(r.keys() for r in rows))))
    w.writeheader();w.writerows(rows);p.write(path,s.getvalue())


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def scene_values(data,key,n=8):
    d=defaultdict(list)
    for r in data['records']:
        if r['steps']==n:d[r['scene_id']].append(r[key])
    assert all(len(x)==4 for x in d.values())
    return {s:float(np.mean(v)) for s,v in d.items()}


def render(root):
    out=root/'final_evaluation';out.mkdir(exist_ok=True)
    audit=json.loads((out/'test_audit.json').read_text());assert audit['status']=='passed'
    ref=Path(json.loads((root/'execution_scope.json').read_text())['reference'])
    p.SOURCES=VARIANTS;p.COLORS=tuple(COLORS[v] for v in VARIANTS)
    data={};old={};summary=[];histories=[];paired=[]
    for v in VARIANTS:
        path=root/'8x8'/v/'test_diagnostics.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['variants'][v]['test_json_sha256']
        data[v]=json.loads(path.read_text());old[v]=json.loads((ref/'8x8'/v/'test_diagnostics.json').read_text())
        for weight,d in [('online',data[v]),('EMA',old[v])]:
            for n in (2,4,8,12):
                row=dict(variant=v,weights=weight,steps=n,scenes=160,realizations=4)
                for k in ('db_rmse','fine_mse','db_bias','horizontal_lag1_gradient_mse'):
                    row[k]=float(np.mean(list(scene_values(d,k,n).values())))
                row['ssim']=float(np.mean([r['ssim'] for r in d['aggregated'] if r['steps']==n]))
                summary.append(row)
        for weight,directory in [('online',root),('EMA',ref)]:
            for r in read_csv(directory/'8x8'/v/'metrics.csv'):
                histories.append(dict(variant=v,weights=weight,epoch=int(r['epoch']),
                    train_loss=float(r['train_loss']),val_db_rmse=float(r['val_db_rmse'])))
        for key in ('db_rmse','fine_mse'):
            a=scene_values(data[v],key);b=scene_values(old[v],key)
            assert a.keys()==b.keys()
            paired.extend(dict(variant=v,metric=key,scene=s,delta=a[s]-b[s]) for s in sorted(a))
    write_csv(out/'test_summary.csv',summary);write_csv(out/'validation_history.csv',histories)
    write_csv(out/'paired_scene_effects.csv',paired)

    fig,axes=p.frame('EMA changes the evaluated trajectory, not the optimizer trajectory',
        'Validation N=2, draw 0; all 1000 epochs; log y-axis; exact training-loss and endpoint equality audited',cols=1)
    ax=axes[0]
    for v in VARIANTS:
        for weight,style in [('online','-'),('EMA','--')]:
            rr=[r for r in histories if r['variant']==v and r['weights']==weight]
            assert all(r['val_db_rmse']>0 for r in rr)
            ax.plot([r['epoch'] for r in rr],[r['val_db_rmse'] for r in rr],
                color=COLORS[v],linestyle=style,label=v+' '+weight,linewidth=1.3)
    ax.set(xlabel='Epoch',ylabel='Global valid-pixel RMSE (dB)',yscale='log');ax.grid(alpha=.15)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9)
    p.save(fig,axes,out/'learning_ema_vs_online',dict(source='validation_history.csv',smoothing='none',
        uncertainty='One training seed; deterministic history, no replicate intervals'))

    p.compare_steps(data,out,'test_euler_comparison','No-EMA final test: pointwise and spatial-detail fidelity')
    fig,axes=p.frame('EMA endpoint penalty is consistent but modest',
        'Each point: one of 160 paired scenes; four draws averaged first; diamond: mean; negative favors online',cols=2)
    for ax,key,title,unit in zip(axes,('db_rmse','fine_mse'),('a  Pointwise error change','b  Detail error change'),('dB','dB²')):
        for i,v in enumerate(VARIANTS):
            values=np.array([r['delta'] for r in paired if r['variant']==v and r['metric']==key])
            ax.scatter(values,i+np.linspace(-.16,.16,len(values)),color=COLORS[v],s=9,alpha=.45,rasterized=True)
            ax.scatter(values.mean(),i,color='black',marker='D',s=32,zorder=5)
        ax.axvline(0,color='#888888',ls=':',lw=.8)
        ax.set(title=title,xlabel=f'Online minus EMA ({unit})',yticks=range(4),yticklabels=VARIANTS,ylim=(-.5,3.5))
    p.save(fig,axes,out/'paired_ema_effect',dict(source='paired_scene_effects.csv',
        uncertainty='Full scene effects shown; no independent-training-seed CI or p-values'))

    records={r['scene_id']:r for r in map(json.loads,(root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl').read_text().splitlines())}
    dataset=Path(json.loads((root/'8x8/CASA/config.json').read_text())['dataset_root'])
    regional=[];wave=[]
    for v in VARIANTS:
        for weight,directory in [('online',root),('EMA',ref)]:
            rc=out/f'regional_{v}_{weight}.csv';wc=out/f'haar_{v}_{weight}.csv'
            if rc.exists() and wc.exists():r,w=read_csv(rc),read_csv(wc)
            else:
                r,w=regional_analysis(directory/'8x8'/v/'test_diagnostics_predictions',v,1000,dataset,records)
                for x in r+w:x['weights']=weight
                write_csv(rc,r);write_csv(wc,w)
            regional+=r;wave+=w
            print('regional complete',v,weight,flush=True)
    groups=defaultdict(lambda:defaultdict(list))
    for r in regional:
        if r['rmse'] not in (None,''):groups[r['variant'],r['weights'],r['region']][r['scene']].append(float(r['rmse']))
    region_summary=[dict(variant=v,weights=weight,region=g,scenes=len(s),rmse=float(np.mean([np.mean(x) for x in s.values()])))
                    for (v,weight,g),s in sorted(groups.items())]
    write_csv(out/'test_regional_summary.csv',region_summary)
    wavegroups=defaultdict(lambda:defaultdict(list))
    for r in wave:
        for metric in ('mse','signed_coefficient_correlation'):
            if r[metric] not in (None,''):
                wavegroups[r['variant'],r['weights'],r['level'],r['orientation'],metric][r['scene']].append(float(r[metric]))
    waves=[dict(variant=v,weights=weight,level=l,orientation=o,metric=m,scenes=len(s),value=float(np.mean([np.mean(x) for x in s.values()])))
           for (v,weight,l,o,m),s in wavegroups.items()]
    write_csv(out/'test_haar_summary.csv',waves)
    fig,axes=p.frame('Errors remain concentrated near buildings',
        'Test N=8; scene-equal RMSE; geometry-only proxy, not verified reflection or diffraction labels',cols=2)
    for ax,g,title in zip(axes,('near_building_le8px','far_building_gt16px'),('a  Near building: distance ≤8 pixels','b  Far: distance >16 pixels')):
        for i,v in enumerate(VARIANTS):
            vals=[next(r['rmse'] for r in region_summary if r['variant']==v and r['weights']==w and r['region']==g) for w in ('EMA','online')]
            ax.plot(vals,[i,i],color=COLORS[v],lw=1.5)
            ax.scatter(vals[0],i,facecolors='none',edgecolors=COLORS[v],s=48)
            ax.scatter(vals[1],i,color=COLORS[v],marker='D',s=34)
        ax.set(title=title,xlabel='Scene-mean RMSE (dB)',yticks=range(4),yticklabels=VARIANTS,ylim=(-.5,3.5))
        ax.grid(axis='x',alpha=.15)
    fig.text(.5,.04,'Open circle: EMA; filled diamond: online. Region membership is identical for paired predictions.',ha='center',fontsize=9)
    p.save(fig,axes,out/'test_regions',dict(source='test_regional_summary.csv',
        uncertainty='Single trained seed; paired point estimates; scene counts in CSV',distance_units='256 benchmark-grid pixels'))
    maps(root,out,'test',Path('test_diagnostics_predictions'),'test_N8')
    report(root,out,summary,region_summary,histories,audit)
    p.write(out/'render_provenance.json',dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helpers={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
            ('no_ema_test_maps.py','no_ema_regional.py','visualize_source_suite.py')},
        audit_sha256=hashlib.sha256((out/'test_audit.json').read_bytes()).hexdigest(),
        source='verified final tests, unchanged training and raw data',formats=['PNG300dpi','PDF','SVG','CSV']))


def report(root,out,summary,regions,histories,audit):
    lines=['# No-EMA experiment: complete test and evaluation','',
        'All four arms completed 1000 epochs; the complete test covers 160 scenes, four source draws, and N=2/4/8/12.',
        'The final last.pt is fixed; the test split does not select a checkpoint or step count. The data had been inspected previously, so this is exploratory evidence.',
        'RMSE and Haar are averaged over four draws within each scene and then equally over scenes; these are neither ensemble-mean-map metrics nor global-pixel RMSE. SSIM is averaged after aggregation over valid windows.','',
        '## 1. Main findings','',
        'All 1000 training-epoch losses, final online weights, and old/new online-weight test records match exactly across the four arms.',
        'Disabling EMA did not produce a new learning trajectory; it exposed capability already learned by the online network but not yet shown by EMA.',
        'Early EMA maps cannot establish whether the network learned the global shape. Final EMA still incurs a smaller error penalty, but it cannot alone explain remaining local-structure gaps.','',
        '## 2. Registered primary endpoint: test N=8','',
        '| Variant | EMA RMSE | Online RMSE | Online-EMA | Online SSIM | Online Haar MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        a=next(r for r in summary if r['variant']==v and r['weights']=='EMA' and r['steps']==8)
        b=next(r for r in summary if r['variant']==v and r['weights']=='online' and r['steps']==8)
        lines.append(f"| {v} | {a['db_rmse']:.4f} | {b['db_rmse']:.4f} | {b['db_rmse']-a['db_rmse']:+.4f} | {b['ssim']:.5f} | {b['fine_mse']:.4f} |")
    lines+=['','CASA has the best test N8 point error, SSIM, and Haar scores; FiLM is close. NSOP slightly improves on ADD but not on CASA or FiLM.',
        'FiLM near-building test RMSE is 13.3132 dB, slightly below CASA at 13.3824 dB; CASA is not best in every region. This small difference needs replication across training seeds.',
        'FiLM and NSOP have matched parameter counts and initial weights, but NSOP invokes the spectral kernel more often per stage. No benefit yet justifies its added complexity.',
        'This evaluates only the present rank-2 shared kernel, fixed modulation, and budget; it does not rule out all condition-driven nonstationary operators.','',
        '## 3. Learning stages: not an endpoint table with the same aggregation','',
        'The following values are per-epoch validation global-valid-pixel RMSE with N2 and source draw 0.','',
        '| Variant | Epoch | EMA RMSE | Online RMSE |','|---|---:|---:|---:|']
    for v in VARIANTS:
        for e in (50,100,200,500,1000):
            vals={w:next(r['val_db_rmse'] for r in histories if r['variant']==v and r['weights']==w and r['epoch']==e) for w in ('EMA','online')}
            lines.append(f"| {v} | {e} | {vals['EMA']:.3f} | {vals['online']:.3f} |")
    lines+=['','The EMA contribution of initial parameters is 0.999^s, where s is the number of successful updates; five updates are planned per epoch.',
        'About 61% of the initial parameter weight remains at epoch 100 and 37% at epoch 200. This parameter-averaging coefficient cannot be converted linearly into prediction error.',
        'Only about 0.67% of the initial contribution remains at the end, but recent parameters are still smoothed by historical parameters, so final performance can lag slightly.','',
        '## 4. Step-count sensitivity: more steps are not always better','',
        '| Variant | Test N2 | Test N4 | Test N8 | Test N12 | Val N4 | Val N8 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    val_rows=[]
    for v in VARIANTS:
        d=json.loads((root/'8x8'/v/'val_diagnostics.json').read_text())
        vals={n:float(np.mean(list(scene_values(d,'db_rmse',n).values()))) for n in (2,4,8,12,32)}
        val_rows.extend(dict(variant=v,steps=n,scene_mean_rmse=value) for n,value in vals.items())
        tests=[next(r['db_rmse'] for r in summary if r['variant']==v and r['weights']=='online' and r['steps']==n) for n in (2,4,8,12)]
        lines.append('| '+v+' | '+' | '.join(f'{x:.4f}' for x in tests+[vals[4],vals[8]])+' |')
    write_csv(out/'val_steps_summary.csv',val_rows)
    lines+=['','All four arms reach lower validation and test RMSE at N4 within the scanned range; the primary comparison retains preregistered N8 rather than selecting from test afterward.',
        'Finer integration follows the learned velocity field and does not guarantee a result closer to ground truth. Velocity-field error interacts with the discretization count.',
        'This curve alone cannot separate velocity bias, trajectory drift, and integration error, nor attribute the effect to FNO K truncation or Gibbs artifacts.','',
        '## 5. Near-building regions remain difficult','',
        '| Variant | Region | Scenes | EMA RMSE | Online RMSE |','|---|---|---:|---:|---:|']
    for v in VARIANTS:
        for g in ('near_building_le8px','far_building_gt16px'):
            a=next(r for r in regions if r['variant']==v and r['weights']=='EMA' and r['region']==g)
            b=next(r for r in regions if r['variant']==v and r['weights']=='online' and r['region']==g)
            lines.append(f"| {v} | {g} | {b['scenes']} | {a['rmse']:.4f} | {b['rmse']:.4f} |")
    lines+=['','Buildings are defined by height > 0; distances are measured in pixels on the current 256 grid and are not converted to meters. Intermediate regions still contribute to overall metrics.',
        'Haar is computed after a common 2x2 benchmark average and retains only fully valid support. dB detail coefficients are not electromagnetic energy or wireless phase.',
        'Near-building errors and fixed crops localize failures, but without path labels they do not verify reflection, diffraction, or Maxwell consistency.','',
        '## 6. Next-step assessment','',
        '- Present online weights as the primary training diagnostic while retaining EMA for side-by-side evaluation; lagging early EMA maps do not negate network capability.',
        '- Retain CASA and FiLM as strong module controls. Current NSOP results do not support its complexity; first test whether conditional modulation locates shadow and enhancement regions.',
        '- Evaluate fine details using signed-coefficient correlation, near-building error, and fixed-scene structure together, rather than texture energy alone.',
        '- One training seed (42); four source draws are not four independent training runs. No claim of cross-configuration generalization or across-seed significance.',
        '- Do not start new training or change the architecture at this stage.','',
        '[Test audit](test_audit.json) | [Complete N-scan CSV](test_summary.csv) | [Regional CSV](test_regional_summary.csv) | [Signed-Haar correlation CSV](test_haar_summary.csv)',
        '\n[Existing loss and gradient plots](../visualizations/README.md) | [Loss and gradient stage analysis](../visualizations/analysis.md)','',
        '## 7. Visualization','',
        'Full maps share -300 to 0 dB; crops share the corresponding GT region 1st-99th percentile color range. This changes display only, not numerical values. Error colors saturate at 30 dB; RMSE is not clipped.',
        'Examples include historically monitored u799 and scenes near the 25th, 50th, and 75th building-coverage percentiles. Crop centers follow height-map edges, not model errors.',
        'Step-count curve error bars show SD across scene means from four source draws, not training-seed uncertainty.','']
    for file in sorted(out.glob('*.png')):
        lines+=['### '+file.stem,'',f'![{file.stem}]({file.name})','',f'[PDF]({file.stem}.pdf) · [SVG]({file.stem}.svg)','']
    p.write(out/'README.md','\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    render(parser.parse_args().root.resolve())
