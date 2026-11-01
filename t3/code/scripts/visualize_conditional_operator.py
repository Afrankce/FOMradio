"""Live diagnostic figures; one training seed, no seed-level uncertainty claim."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.monitor_matern_suite import committed_history,atomic
SKILL=Path(os.environ.get('FIGURE_SKILL_ROOT',str(Path.home()/'.codex/skills/nature-figure')))
sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

VARIANTS=('CASA','ADD','FILM','NSOP')
COLORS={'CASA':'#777777','ADD':'#6B8C67','FILM':'#3975A4','NSOP':'#BC6C45'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})


def csv_out(path,rows):
    if not rows:return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,keys);w.writeheader();w.writerows(rows)


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
        result=subprocess.run([sys.executable,str(SKILL/'scripts'/script),str(path)+'.pdf',*args],
                              capture_output=True,text=True)
        audits[script]=dict(exit_code=result.returncode,output=result.stdout+result.stderr)
    atomic(Path(str(path)+'.qa.json'),audits)
    if any(a['exit_code']!=0 for a in audits.values()):
        raise RuntimeError(f'Figure QA requires review: {path.name}')


def line_plot(out,name,title,ylabel,series,zero=False):
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    fig.subplots_adjust(left=.14,right=.96,bottom=.18,top=.74)
    for label,x,y,color,style in series:
        if len(x):ax.plot(x,y,label=label,color=color,linestyle=style,linewidth=1.3,marker='.' if len(x)<3 else None)
    ax.set(xlabel='Epoch',ylabel=ylabel)
    fig.suptitle(title,y=.97,fontsize=11)
    if series:fig.legend(loc='upper center',bbox_to_anchor=(.55,.92),ncol=min(4,len(series)),fontsize=8)
    if zero:ax.axhline(0,color='#999999',linewidth=.6,linestyle=':')
    ax.grid(alpha=.18)
    export(fig,out/name)


def prediction_stages(directory):
    result={}
    for p in (directory/'previews').glob('epoch_*.json'):
        result[int(p.stem.split('_')[1])]=p.with_name(p.stem+'_predictions')
    for p in (directory/'milestones').glob('epoch_*/val.json'):
        result[int(p.parent.name.split('_')[1])]=p.parent/'val_predictions'
    return result


def mean2(a):return (a[::2,::2]+a[1::2,::2]+a[::2,1::2]+a[1::2,1::2])/4


def regional_analysis(stage,variant,epoch,dataset,manifest):
    rows=[];wave=[]
    for file in sorted(stage.glob('N*/seed_*/*.npz')):
        scene=file.stem;sample=np.load(file,allow_pickle=False)
        mask=sample['valid_mask'][0].astype(bool);target=sample['target'][0]*300-300
        raw=sample['prediction'][0];pred=np.clip(raw,0,1)*300-300
        height=np.load(dataset/manifest[scene]['height_path'],allow_pickle=False)
        if height.shape!=mask.shape:raise ValueError('Height/benchmark grid mismatch')
        building=height>0
        dist=distance_transform_edt(~building) if building.any() else np.full(height.shape,np.inf)
        identity=dict(variant=variant,epoch=epoch,scene=scene,steps=int(file.parent.parent.name[1:]),realization=int(file.parent.name[5:]))
        for name,region in [('all',mask),('near_building_le8px',mask&(dist<=8)),('far_building_gt16px',mask&(dist>16))]:
            error=(pred-target)[region]
            rows.append(dict(**identity,region=name,count=int(region.sum()),
                             rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                             bias=float(np.mean(error)) if len(error) else None))
        p,y,valid=mean2(pred),mean2(target),mean2(mask.astype(float))==1
        for level in (1,2):
            support=mean2(valid.astype(float))==1
            def haar(a):
                aa,bb,cc,dd=a[::2,::2],a[::2,1::2],a[1::2,::2],a[1::2,1::2]
                return (aa+bb+cc+dd)/2,[(aa-bb+cc-dd)/2,(aa+bb-cc-dd)/2,(aa-bb-cc+dd)/2]
            p,pd=haar(p);y,yd=haar(y)
            for orientation,a,b in zip(('column_difference','row_difference','diagonal_difference'),pd,yd):
                av,bv=a[support],b[support];count=len(av)
                corr=float(np.corrcoef(av,bv)[0,1]) if count>1 and av.std()>0 and bv.std()>0 else None
                wave.append(dict(**identity,level=level,orientation=orientation,count=count,
                    mse=float(np.mean((av-bv)**2)) if count else None,
                    pred_coefficient_mean_square=float(np.mean(av**2)) if count else None,
                    target_coefficient_mean_square=float(np.mean(bv**2)) if count else None,
                    signed_coefficient_correlation=corr))
            valid=support
    return rows,wave


def run(root):
    out=root/'visualizations';out.mkdir(exist_ok=True)
    scope=json.loads((root/'execution_scope.json').read_text())
    directories={v:root/'8x8'/v for v in VARIANTS}
    histories={v:committed_history(d)[0] for v,d in directories.items()}
    diagnostic={};flat=[];parameter=[]
    for v in VARIANTS:
        committed=histories[v][-1]['epoch'] if histories[v] else 0
        rows=[]
        for file in sorted((directories[v]/'loss_diagnostics').glob('window_*.json')):
            r=json.loads(file.read_text())
            if r['epoch_index']>=committed:continue
            r['epoch']=r['epoch_index']+1;rows.append(r)
            flat.append(dict(variant=v,epoch=r['epoch'],step=r['optimizer_step_before'],**r['losses'],
                             auxiliary=r['auxiliary'],total=r['total'],optimizer_ran=r['optimizer_ran'],
                             **{'output_'+k:val for k,val in r['output_gradients'].items()}))
            if r['parameter_gradients']:
                for group,values in r['parameter_gradients'].items():
                    parameter.append(dict(variant=v,epoch=r['epoch'],step=r['optimizer_step_before'],group=group,**values))
        diagnostic[v]=rows
    csv_out(out/'loss_windows.csv',flat);csv_out(out/'parameter_gradients.csv',parameter)
    csv_out(out/'validation_history.csv',[dict(variant=v,**r) for v,rows in histories.items() for r in rows])
    if not any(diagnostic.values()):
        atomic(out/'README.md','# Loss diagnostics\n\nWaiting for the first committed training epoch.\n');return
    def epoch_series(fn):
        series=[]
        for v,rows in diagnostic.items():
            epochs=sorted({r['epoch'] for r in rows});ys=[]
            for e in epochs:
                vals=[fn(r) for r in rows if r['epoch']==e];vals=[x for x in vals if x is not None]
                ys.append(float(np.mean(vals)) if vals else np.nan)
            series.append((v,epochs,ys,COLORS[v],'-'))
        return series
    line_plot(out,'fm_loss','Main objective: valid-pixel velocity MSE','FM MSE',epoch_series(lambda r:r['losses']['fm']))
    line_plot(out,'weighted_auxiliary','Auxiliary objective contribution (weight = 0.1)','Weighted auxiliary loss',epoch_series(lambda r:r['weight']*r['auxiliary']))
    line_plot(out,'output_gradient_ratio','Output-gradient contribution','Weighted auxiliary / FM norm',epoch_series(lambda r:r['output_gradients']['weighted_ratio']))
    for metric,title,label in [('weighted_ratio','Parameter-gradient contribution','Weighted auxiliary / FM norm'),
                               ('cosine','Main vs auxiliary parameter-gradient alignment','Cosine similarity')]:
        series=[]
        for v in VARIANTS:
            rows=[r for r in parameter if r['variant']==v and r['group']=='all' and r[metric] is not None]
            series.append((v,[r['epoch'] for r in rows],[r[metric] for r in rows],COLORS[v],'-'))
        line_plot(out,'parameter_'+metric,title,label,series,zero=metric=='cosine')
    series=[]
    for v in VARIANTS:
        rows=[r for r in parameter if r['variant']==v and r['group']=='all']
        for key,style,label in [('fm_norm','-','FM'),('weighted_aux_norm','--','Aux (0.1)')]:
            series.append((v+' '+label,[r['epoch'] for r in rows],[r[key] for r in rows],COLORS[v],style))
    line_plot(out,'parameter_norms','Parameter-gradient strength before Adam','Gradient norm',series)
    group_colors=['#3975A4','#BC6C45','#6B8C67','#8B6FA5','#777777','#A88A42']
    for v in VARIANTS:
        series=[]
        for group,color in zip(('spectral','local','attention','condition_encoder','other','conditional_operator'),group_colors):
            rows=[r for r in parameter if r['variant']==v and r['group']==group and r['weighted_ratio'] is not None]
            series.append((group.replace('_',' ').capitalize(),[r['epoch'] for r in rows],[r['weighted_ratio'] for r in rows],color,'-'))
        line_plot(out,v+'_module_gradient_ratio',v+': auxiliary influence by module','Weighted auxiliary / FM norm',series)
    max_epoch=max([r['epoch'] for v,rows in histories.items()  for r in rows] or [0])
    series=[]
    for v,rows in histories.items():
        rows=[r for r in rows if r['epoch']<=max_epoch]
        series.append((v,[r['epoch'] for r in rows],[r['val_db_rmse'] for r in rows],COLORS[v],'-'))
    line_plot(out,'validation_rmse','Matched-epoch validation (Euler N = 2)','RMSE (dB)',series)
    stages={v:prediction_stages(d) for v,d in directories.items()}
    common=set.intersection(*(set(stages[v]) for v in VARIANTS))
    if common:
        epoch=max(common)
        manifest={r['scene_id']:r for r in map(json.loads,(root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl').read_text().splitlines())}
        # Dataset location is an existing immutable execution input, not a plot assumption.
        cfg=json.loads((directories['CASA']/'config.json').read_text());dataset=Path(cfg['dataset_root'])
        regional=[];wave=[]
        for v in directories:
            if epoch not in stages[v]:continue
            r,w=regional_analysis(stages[v][epoch],v,epoch,dataset,manifest);regional+=r;wave+=w
        csv_out(out/f'regional_epoch_{epoch:04d}.csv',regional)
        csv_out(out/f'haar_orientation_epoch_{epoch:04d}.csv',wave)
        files=sorted((stages['CASA'][epoch]/'N8/seed_0').glob('*.npz'))
        # Four fixed lexicographic validation scenes; no error-based selection.
        for file in files[:4]:
            scene=file.stem
            for v in VARIANTS:
                other=stages[v][epoch]/'N8/seed_0'/file.name
                if not other.exists():continue
                sample=np.load(other,allow_pickle=False);mask=sample['valid_mask'][0].astype(bool)
                target=sample['target'][0]*300-300;pred=np.clip(sample['prediction'][0],0,1)*300-300
                fig,axes=plt.subplots(1,3,figsize=(11.2,3.6))
                fig.subplots_adjust(left=.03,right=.97,bottom=.17,top=.78,wspace=.2)
                for ax,field,title,cmap,lo,hi in zip(axes,[target,pred,np.abs(pred-target)],
                        ['Ground truth (dB)',v+' prediction (dB)','Absolute error (dB)'],['viridis','viridis','magma'],[-300,-300,0],[0,0,100]):
                    cm=plt.get_cmap(cmap).copy();cm.set_bad('#b5b5b5')
                    ax.imshow(np.ma.array(field,mask=~mask),cmap=cm,vmin=lo,vmax=hi)
                    ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
                fig.suptitle(f'{scene} | Epoch {epoch} | Euler N=8 | Fixed source draw 0',fontsize=11,y=.97)
                fig.text(.5,.05,'Map colors: -300 to 0 dB; error colors: 0 to 100 dB (saturated above); gray: invalid',ha='center',fontsize=8)
                export(fig,out/f'map_{v}_{scene}_epoch_{epoch:04d}')
    images=sorted(p.name for p in out.glob('*.png'))
    atomic(out/'README.md','# Conditional operator live diagnostics\n\n'
        'One training seed (42). Curves are descriptive, not independent-seed uncertainty. '
        'All four models are fresh matched-budget runs; shared MS source and FM + 0.1 Haar loss. '
        'Parameter gradients are global, raw and unscaled, before Adam; not spatial derivatives. '
        'Loss curves average batch means within each epoch; support counts remain in raw JSON. '
        'Haar orientation labels describe array differences, not RF propagation mechanisms.\n\n'+
        '\n\n'.join(f'![{name}]({name})' for name in images))
    atomic(out/'render_provenance.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figures=images,source='committed epochs only; per-window JSON and saved validation predictions',
        uncertainty='one training seed; no independent-seed intervals',export='PNG 300 dpi, PDF, SVG; reporting canvas'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root.resolve())
