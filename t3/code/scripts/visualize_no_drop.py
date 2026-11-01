"""Matched-epoch AP dropout ablation, numerical predictions only.

Structural reuse of the audited project matplotlib export/curve helpers.
One training seed; four source draws are NOT independent training repetitions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.visualize_spectral_gate import (plt,line_plot,csv_out,prediction_stages,
    regional_analysis,committed_history,atomic,SKILL,require_matplotlib_panel_alignment)

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                     'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})


def export(fig,path):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',strict=True)
    fig.savefig(str(path)+'.png',dpi=300)
    fig.savefig(str(path)+'.pdf')
    fig.savefig(str(path)+'.svg')
    plt.close(fig)
    audits={}
    for script,args in [('audit_pdf_text.py',['--min-pt','5']),
        ('audit_figure_collisions.py',['--json-out',str(path)+'.collision.json'])]:
        r=subprocess.run([sys.executable,str(SKILL/'scripts'/script),str(path)+'.pdf',*args],capture_output=True,text=True)
        audits[script]=dict(exit_code=r.returncode,output=r.stdout+r.stderr)
    atomic(Path(str(path)+'.qa.json'),audits)
    if any(r['exit_code'] for r in audits.values()):raise RuntimeError(f'Figure QA failed: {path}')

COLORS={'Drop025':'#777777','NoDrop':'#3975A4'}


def compare_evaluations(a,b,epoch,split):
    identity_keys=('source','epoch','steps','realizations')
    for k in identity_keys:
        if a['identity'][k]!=b['identity'][k]:raise ValueError(f'Unmatched evaluation {k}')
    key=lambda r:(r['scene_id'],r['steps'],r['realization'])
    aa={key(r):r for r in a['records']};bb={key(r):r for r in b['records']}
    if aa.keys()!=bb.keys():raise ValueError('Unmatched scene/draw records')
    per_scene=[]
    for n in sorted({k[1] for k in aa}):
        scenes=sorted({k[0] for k in aa if k[1]==n})
        for s in scenes:
            keys=[k for k in aa if k[0]==s and k[1]==n]
            row=dict(scene=s,epoch=epoch,split=split,steps=n,draws=len(keys))
            for field in ('db_rmse','db_bias','fine_mse'):
                for name,d in [('Drop025',aa),('NoDrop',bb)]:
                    row[name+'_'+field]=float(np.mean([d[k][field] for k in keys]))
                row['delta_'+field]=row['NoDrop_'+field]-row['Drop025_'+field]
            per_scene.append(row)
    summary=[]
    for n in sorted({k[1] for k in aa}):
        for label,d in [('Drop025',a),('NoDrop',b)]:
            rs=[r for r in d['records'] if r['steps']==n];ag=[r for r in d['aggregated'] if r['steps']==n]
            summary.append(dict(epoch=epoch,split=split,steps=n,model=label,n_scenes=d['n_scenes'],
                draws=d['identity']['realizations'],
                pooled_rmse_db=float(np.sqrt(np.average([r['db_rmse']**2 for r in rs],weights=[r['valid_pixels'] for r in rs]))),
                ssim=float(np.mean([r['ssim'] for r in ag])),
                bias_db=float(np.average([r['db_bias'] for r in rs],weights=[r['valid_pixels'] for r in rs])),
                fine_mse=float(np.average([r['fine_mse'] for r in rs],weights=[r['fine_count'] for r in rs]))))
    return per_scene,summary


def run(root):
    out=root/'visualizations';out.mkdir(exist_ok=True)
    scope=json.loads((root/'execution_scope.json').read_text())
    dirs={'Drop025':Path(scope['reference_run']),'NoDrop':root/'AP'}
    hist={k:committed_history(d)[0] for k,d in dirs.items()}
    if not hist['NoDrop']:return
    end=hist['NoDrop'][-1]['epoch'];hist={k:[r for r in rs if r['epoch']<=end] for k,rs in hist.items()}
    csv_out(out/'validation_history.csv',[dict(model=k,**r) for k,rs in hist.items() for r in rs])
    for metric,name,title,ylabel in [('val_db_rmse','validation_rmse','Full-condition validation | Euler N2','RMSE (dB)'),
        ('train_loss','training_fm','Training FM loss | Different tasks: mixed vs conditioned','Velocity MSE')]:
        line_plot(out,name,title,ylabel,[(k,[r['epoch'] for r in rs],[r[metric] for r in rs],COLORS[k],'-') for k,rs in hist.items()])
    windows=[];params=[]
    for k,d in dirs.items():
        for p in sorted((d/'loss_diagnostics').glob('window_*.json')):
            r=json.loads(p.read_text())
            if r['epoch_index']>=end:continue
            windows.append(dict(model=k,epoch=r['epoch_index']+1,step=r['optimizer_step_before'],
                **r['losses'],auxiliary=r['auxiliary'],total=r['total'],optimizer_ran=r['optimizer_ran']))
            for group,v in (r['parameter_gradients'] or {}).items():
                params.append(dict(model=k,epoch=r['epoch_index']+1,group=group,**v))
    csv_out(out/'training_windows.csv',windows);csv_out(out/'parameter_gradients.csv',params)
    for metric,title,label in [('cosine','FM vs Haar: shared-parameter gradient alignment','Cosine similarity'),
        ('weighted_ratio','Relative Haar contribution to parameter gradients','0.1 Haar / FM gradient norm')]:
        series=[]
        for k in dirs:
            rs=[r for r in params if r['model']==k and r['group']=='all' and r[metric] is not None]
            series.append((k,[r['epoch'] for r in rs],[r[metric] for r in rs],COLORS[k],'-'))
        line_plot(out,'gradient_'+metric,title,label,series,zero=metric=='cosine')
    # Original group naming puts embed_model under "other"; never relabel it as encoder-only.
    stages={k:prediction_stages(d) for k,d in dirs.items()}
    common=sorted(set(stages['Drop025'])&set(stages['NoDrop']))
    manifest={r['scene_id']:r for r in map(json.loads,(root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl').read_text().splitlines())}
    cfg=json.loads((dirs['Drop025']/'config.json').read_text());dataset=Path(cfg['dataset_root'])
    summary=[];paired=[]
    for e in common:
        paths={k:(d/'milestones'/f'epoch_{e:04d}'/'val.json' if e%100==0 else d/'previews'/f'epoch_{e:04d}.json') for k,d in dirs.items()}
        a,b=[json.loads(paths[k].read_text()) for k in dirs]
        pairs,stats=compare_evaluations(a,b,e,'val');paired+=pairs;summary+=stats
        # Save regional and signed Haar diagnostics for every observed stage once.
        regional_file=out/f'regional_epoch_{e:04d}.csv'
        if not regional_file.exists():
            reg=[];wave=[]
            for k in dirs:
                r,w=regional_analysis(stages[k][e],k,e,dataset,manifest);reg+=r;wave+=w
            csv_out(regional_file,reg);csv_out(out/f'haar_epoch_{e:04d}.csv',wave)
    if all((d/'test_diagnostics.json').exists() for d in dirs.values()):
        a,b=[json.loads((d/'test_diagnostics.json').read_text()) for d in dirs.values()]
        pairs,stats=compare_evaluations(a,b,1000,'test');paired+=pairs;summary+=stats
        ci=[]
        for n in (4,8):
            rs=[r for r in pairs if r['steps']==n]
            rng=np.random.default_rng(20260911);ix=rng.integers(len(rs),size=(2000,len(rs)))
            for metric in ('db_rmse','fine_mse'):
                delta=np.array([r['delta_'+metric] for r in rs]);boot=delta[ix].mean(1)
                lo,hi=np.quantile(boot,[.025,.975])
                ci.append(dict(steps=n,metric=metric,mean_scene_difference=float(delta.mean()),low=float(lo),high=float(hi),n_scenes=len(rs),
                    definition='NoDrop minus Drop025; scene bootstrap, source draws averaged within scene; single training seed'))
        atomic(out/'test_paired_intervals.json',ci)
    csv_out(out/'paired_scenes.csv',paired);csv_out(out/'evaluation_summary.csv',summary)
    # N8 whole-validation curves exclude the early four-scene preview statistics.
    if any(r['n_scenes']==80 for r in summary):
        series=[]
        for k in dirs:
            rs=[r for r in summary if r['model']==k and r['split']=='val' and r['steps']==8 and r['n_scenes']==80]
            series.append((k,[r['epoch'] for r in rs],[r['pooled_rmse_db'] for r in rs],COLORS[k],'-'))
        line_plot(out,'validation_N8_rmse','Full validation | N8, four paired source draws','Pooled RMSE (dB)',series)
    map_jobs=[]
    if common:
        e=common[-1];files=sorted((stages['Drop025'][e]/'N8/seed_0').glob('*.npz'))[:4]
        map_jobs += [(p.stem,e,'val',{k:stages[k][e]/'N8/seed_0'/p.name for k in dirs}) for p in files]
    if all((d/'test_diagnostics.json').exists() for d in dirs.values()):
        for s in ('u4','u406'):
            map_jobs.append((s,1000,'test',{k:d/'test_diagnostics_predictions/N8/seed_0'/f'{s}.npz' for k,d in dirs.items()}))
    for scene,e,split,paths in map_jobs:
        name=f'{split}_{scene}_epoch_{e:04d}'
        if (out/(name+'.qa.json')).exists():continue
        data={k:dict(np.load(p,allow_pickle=False)) for k,p in paths.items()}
        for field in ('target','valid_mask'):
            if not np.array_equal(data['Drop025'][field],data['NoDrop'][field]):raise ValueError('Unmatched map')
        valid=data['NoDrop']['valid_mask'][0].astype(bool);y=data['NoDrop']['target'][0]*300-300
        preds={k:np.clip(v['prediction'][0],0,1)*300-300 for k,v in data.items()}
        fig,axes=plt.subplots(1,5,figsize=(15,3.7));fig.subplots_adjust(left=.025,right=.98,bottom=.18,top=.76,wspace=.16)
        fields=[y,preds['Drop025'],preds['NoDrop'],abs(preds['Drop025']-y),abs(preds['NoDrop']-y)]
        titles=['Ground truth','AP: Drop 0.25','AP: No-drop','Drop error','No-drop error']
        for i,(ax,f,title) in enumerate(zip(axes,fields,titles)):
            cm=plt.colormaps['jet' if i<3 else 'magma'].copy();cm.set_bad('black')
            ax.imshow(np.ma.array(f,mask=~valid),cmap=cm,vmin=-300 if i<3 else 0,vmax=0 if i<3 else 100,interpolation='nearest')
            ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'{scene} | {split} | Matched epoch {e} | Online, N8, source draw 0',y=.97,fontsize=12)
        fig.text(.5,.08,'Radio: jet, -300 to 0 dB. Error: magma, 0 to 100 dB (saturated above). Black: building / invalid.',ha='center',fontsize=9)
        export(fig,out/name)
        atomic(out/(name+'.inputs.json'),dict(source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
            selection='first four available validation scenes; u4/u406 fixed historical user-selected test scenes',scene=scene,epoch=e,split=split))
    probes=[]
    for p in sorted((root/'monitoring/probes').glob('epoch_*.json')):probes+=json.loads(p.read_text())['rows']
    if probes:
        series=[]
        for k in dirs:
            rs=[r for r in probes if r['model']==k and r['time']==0.]
            series.append((k,[r['epoch'] for r in rs],[r['shared_conditional_null_cosine'] for r in rs],COLORS[k],'-'))
        line_plot(out,'conditional_null_gradient','Diagnostic at t=0 | Four fixed validation scenes','Conditional vs null gradient cosine',series,zero=True)
        atomic(out/'conditional_null_gradients.json',probes)
    imgs=sorted(p.name for p in out.glob('*.png'))
    atomic(out/'README.md','# AP No-drop: paired learning and spatial-structure visualizations\n\n'+
        f'Observed through epoch {end}. The comparator uses original AP dropout=0.25; the new experiment uses dropout=0. Other conditions are fixed.\n\n'+
        'One training seed (42); no significance claim across training randomness. Training losses have different tasks, mixed conditional/unconditional versus fully conditional, and cannot be ranked directly.\n\n'+
        'N2 is fully conditional validation each epoch; N8 uses full validation every 100 epochs, 80 scenes x four draws. Epochs 1/10/50 have four-scene previews only and are not mixed into the N8 aggregate curve.\n\n'+
        'Gradients are parameter derivatives, not spatial gradients. FM/Haar gradients are logged every 25 steps. Conditional/null probes are independent and read-only; null input is out of distribution for No-drop.\n\n'+
        'Regional metrics use a building-distance proxy, not ray-traced LoS labels. The original gradient group other includes the condition encoder and is not renamed to an encoder-only gradient.\n\n'+
        'Color limits are fixed; export is PNG at 300 dpi, PDF, and SVG. Canvas sizes are for reporting, not journal requirements. Automated font/collision/alignment checks were run; later generated figures still require manual review.\n\n'+
        '[Paired metrics](evaluation_summary.csv) · [Scene differences](paired_scenes.csv) · [Training detail](training_windows.csv) · [Gradient detail](parameter_gradients.csv)\n\n'+
        '\n\n'.join(f'![{name}]({name})' for name in imgs))
    atomic(out/'render_provenance.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        epoch=end,figures=imgs,backend='matplotlib',uncertainty='one training seed; only final scene-bootstrap intervals',
        exclusions='reference curves cropped to observed NoDrop epoch; early 4-scene previews excluded from 80-scene N8 aggregate curve'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root.resolve())
