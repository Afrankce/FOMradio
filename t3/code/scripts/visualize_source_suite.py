"""Source-only study: priors, live convergence, matched milestones and final maps."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SOURCES=('W','S2','S8','S32','MS')
COLORS=('#465363','#4c82aa','#cf9557','#57968a','#9365a4')
QA=Path('/home/ka/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(QA))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':10,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False) if not isinstance(value,str) else value)
    temporary.replace(path)


def ready(path):
    metadata=path.with_suffix('.json')
    if not metadata.exists(): return False
    try: return all(c['exit_code']==0 for c in json.loads(metadata.read_text())['qa'])
    except (KeyError,ValueError): return False


def save(fig,axes,path,metadata):
    path.parent.mkdir(parents=True,exist_ok=True);fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=list(np.asarray(axes).flat),json_out=path.with_suffix('.alignment.json'),strict=True)
    for suffix in ('png','pdf','svg'):
        fig.savefig(path.with_suffix('.'+suffix),dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    checks=[]
    for script,args in [('audit_pdf_text.py',['--min-pt','5','--json']),
                         ('audit_figure_collisions.py',['--json-out',str(path.with_suffix('.collision.json'))])]:
        result=subprocess.run([sys.executable,str(QA/script),str(path.with_suffix('.pdf')),*args],capture_output=True,text=True)
        checks.append(dict(script=script,exit_code=result.returncode,output=result.stdout,stderr=result.stderr))
    write(path.with_suffix('.json'),dict(**metadata,qa=checks))
    if any(c['exit_code'] for c in checks):raise RuntimeError(f'Figure QA failed: {path}')


def frame(title,subtitle,cols=3,height=4.7):
    fig,axes=plt.subplots(1,cols,figsize=(14,height),squeeze=False)
    fig.subplots_adjust(left=.065,right=.985,bottom=.21,top=.76,wspace=.32)
    fig.suptitle(title,y=.98,fontsize=14,fontweight='bold')
    fig.text(.5,.89,subtitle,ha='center',fontsize=10)
    return fig,axes[0]


def priors(root,out):
    path=out/'source_fields'
    if ready(path) and ready(out/'source_covariance'):return
    z=np.load(root/'source_examples.npz',allow_pickle=False)
    fig,axes=plt.subplots(1,5,figsize=(14,3.8))
    fig.subplots_adjust(left=.025,right=.975,top=.79,bottom=.22,wspace=.12)
    fig.suptitle('External sources: one paired realization, before training',y=.99,fontsize=14,fontweight='bold')
    for i,(ax,name) in enumerate(zip(axes,SOURCES)):
        im=ax.imshow(z[name],cmap='RdBu_r',vmin=-3,vmax=3,interpolation='nearest');ax.set_title(f'{chr(97+i)}  {name}');ax.set_axis_off()
    cb=fig.colorbar(im,cax=fig.add_axes([.30,.13,.40,.03]),orientation='horizontal',extend='both')
    cb.set_label('Source value (shared scale; display saturates outside ±3)',fontsize=9)
    save(fig,axes,path,dict(role='sampler visualization, not model predictions',seed_namespace='source_preview',
        source_sha256=hashlib.sha256((root/'source_examples.npz').read_bytes()).hexdigest(),normalization='none',sample_count=1))
    fig,axes=frame('Source covariance and spectral support','Analytic discrete covariance / embedding eigenvalues; no target fitting',cols=2)
    for name,color in zip(SOURCES,COLORS):
        axes[0].plot(np.arange(129),z[name+'_covariance'],label=name,color=color)
        spectrum=z[name+'_spectrum'];n=spectrum.shape[0]
        # A common axis slice of the actual circulant embedding, not a radial PSD estimate.
        axes[1].loglog(np.fft.rfftfreq(n)[1:],spectrum[0,1:],color=color,label=name)
    axes[0].set(title='a  Spatial dependence',xlabel='Separation (source-grid pixels)',ylabel='Covariance')
    axes[1].set(title='b  Spectral support',xlabel='Spatial frequency (cycles / source pixel)',ylabel='Embedding eigenvalue')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5)
    save(fig,axes,out/'source_covariance',dict(role='source specification',spectral_slice='ky=0, kx>0; not radial average'))


def curves(root,out):
    histories={}
    for name in SOURCES:
        directory=root/'8x8'/name
        if not (directory/'committed_epoch.json').exists():continue
        committed=json.loads((directory/'committed_epoch.json').read_text())['completed_epochs']
        with (directory/'metrics.csv').open() as stream:rows=list(csv.DictReader(stream))
        rows=[r for r in rows if int(r['epoch'])<=committed]
        if rows:histories[name]=rows
    if not histories:return
    fig,axes=frame('Source-only training progress','All committed epochs; seed 42; latest endpoints may have different training budgets')
    for name,color in zip(SOURCES,COLORS):
        if name not in histories:continue
        rows=histories[name];epochs=[int(r['epoch']) for r in rows]
        for ax,key in zip(axes,('train_loss','val_db_rmse','val_ssim')):
            ax.plot(epochs,[float(r[key]) for r in rows],label=name,color=color,lw=1.5)
    axes[0].set_yscale('log')
    for ax,title,ylabel in zip(axes,('a  Optimization','b  Pointwise validation','c  Structural validation'),
                              ('Masked velocity MSE','Validation RMSE (dB)','Validation SSIM')):
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel);ax.grid(axis='y',alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5)
    save(fig,axes,out/'training_curves',dict(last_epochs={k:int(v[-1]['epoch']) for k,v in histories.items()},
        validation='N2, realization0, CFG1',uncertainty='one training seed; deterministic history, no CI',smoothing='none'))


def compare_steps(data,out,name,title):
    fig,axes=frame(title,'Same epoch / checkpoint; mean ± SD over 4 realizations, not independent training seeds')
    numeric=[]
    for source,color in zip(SOURCES,COLORS):
        rows=data[source]['records'];steps=sorted({r['steps'] for r in rows})
        for ax,key in zip(axes,('db_rmse','fine_mse','horizontal_lag1_gradient_mse')):
            values=np.array([[np.mean([r[key] for r in rows if r['steps']==n and r['realization']==s]) for s in range(4)] for n in steps])
            ax.errorbar(steps,values.mean(1),yerr=values.std(1,ddof=1),label=source,color=color,marker='o',capsize=3,markersize=4)
            numeric.append(dict(source=source,metric=key,steps=steps,mean=values.mean(1).tolist(),sd=values.std(1,ddof=1).tolist()))
    for ax,title,ylabel in zip(axes,('a  Pointwise fidelity','b  Fine-detail fidelity','c  Local gradient fidelity'),
                              ('Scene-mean RMSE (dB)','Haar detail MSE (dB²)','Horizontal gradient MSE (dB²)')):
        ax.set(title=title,xlabel='Euler steps / NFE',ylabel=ylabel);ax.grid(axis='y',alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5)
    save(fig,axes,out/name,dict(values=numeric,n_scenes={k:v['n_scenes'] for k,v in data.items()},
        uncertainty='SD of four per-realization scene means; one trained seed'))


def maps(root,out,split,subdirectory,tag,steps=8,scenes=None):
    manifest=[json.loads(x) for x in (root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl').read_text().splitlines()]
    records={r['scene_id']:r for r in manifest if r['split']==split}
    config=json.loads((root/'8x8/W/config.json').read_text());dataset=Path(config['dataset_root'])
    coverage=sorted((float(np.mean(np.load(dataset/r['radiomap_path'],allow_pickle=False)==1000)),s) for s,r in records.items())
    chosen=['u799'] if split=='test' else []
    for q in (.25,.5,.75):
        s=coverage[round(q*(len(coverage)-1))][1]
        if s not in chosen:chosen.append(s)
    if scenes is not None:chosen=list(scenes)
    cmap=plt.colormaps['viridis'].copy();cmap.set_bad('#b0b0b0')
    errmap=plt.colormaps['magma'].copy();errmap.set_bad('#b0b0b0')
    selections=[]
    for scene in chosen:
        stored={v:np.load(root/'8x8'/v/subdirectory/f'N{steps}/seed_0'/f'{scene}.npz',allow_pickle=False) for v in SOURCES}
        gt=stored['W']['target'][0]*300-300;mask=stored['W']['valid_mask'][0]
        for v in SOURCES:
            assert np.array_equal(stored[v]['target'],stored['W']['target'])
            assert np.array_equal(stored[v]['valid_mask'],stored['W']['valid_mask'])
        pred={v:stored[v]['prediction'][0].clip(0,1)*300-300 for v in SOURCES}
        height=np.load(dataset/records[scene]['height_path'],allow_pickle=False)
        gy,gx=np.gradient(height);edges=np.argwhere(np.hypot(gx,gy)>0)
        center=edges[np.argmin(((edges-[127,127])**2).sum(1))] if len(edges) else np.array([127,127])
        top,left=np.clip(center-32,0,192).astype(int)
        selections.append(dict(scene=scene,crop_rc=[int(top),int(left),64,64]))
        for cropped in (False,True):
            name=f'{tag}_{scene}'+('_crop' if cropped else '_full')
            if ready(out/name):continue
            roi=(slice(top,top+64),slice(left,left+64)) if cropped else (slice(None),slice(None))
            good=mask[roi];truth=gt[roi]
            low,high=np.quantile(truth[good],[.01,.99]) if cropped and good.any() else (-300,0)
            if high<=low:low,high=-300,0
            fig,axes=plt.subplots(2,1+len(SOURCES),figsize=(16,6.5))
            fig.subplots_adjust(left=.025,right=.975,bottom=.20,top=.85,wspace=.10,hspace=.28)
            fig.suptitle(f'{scene} | {tag} | N={steps}, realization 0'+(' | fixed geometry crop' if cropped else ''),y=.97,fontsize=14,fontweight='bold')
            im=axes[0,0].imshow(np.ma.array(truth,mask=~good),vmin=low,vmax=high,cmap=cmap,interpolation='nearest')
            axes[0,0].set_title('Ground truth')
            axes[1,0].imshow(height[roi],cmap='gray',vmin=0,vmax=max(1.,float(height.max())),interpolation='nearest');axes[1,0].set_title('Height (within-scene scale)')
            for col,v in enumerate(SOURCES,1):
                p=pred[v][roi]
                axes[0,col].imshow(np.ma.array(p,mask=~good),vmin=low,vmax=high,cmap=cmap,interpolation='nearest');axes[0,col].set_title(v)
                error=np.abs(p-truth)
                ei=axes[1,col].imshow(np.ma.array(error,mask=~good),vmin=0,vmax=30,cmap=errmap,interpolation='nearest')
                rmse=np.sqrt(np.mean((p[good]-truth[good])**2)) if good.any() else None
                axes[1,col].set_title(f'RMSE {rmse:.2f} dB' if rmse is not None else 'No valid support')
            for ax in axes.flat:ax.set_axis_off()
            if not cropped:
                for ax in axes[0]:ax.add_patch(Rectangle((left-.5,top-.5),64,64,fill=False,edgecolor='white',lw=.8))
            cb=fig.colorbar(im,cax=fig.add_axes([.16,.105,.28,.025]),orientation='horizontal',extend='both' if cropped else 'neither')
            cb.set_label('Radio map (dB); shared contrast',fontsize=9)
            cb=fig.colorbar(ei,cax=fig.add_axes([.60,.105,.28,.025]),orientation='horizontal',extend='max')
            cb.set_label('Absolute error (dB); >30 saturated',fontsize=9)
            save(fig,axes,out/name,dict(scene=scene,split=split,steps=steps,realization=0,crop=selections[-1],
                normalization='display only; source arrays untouched',radio_limits=[float(low),float(high)],
                height_max=float(height.max()),selection='u799 requested case plus geometry coverage quartiles',
                masks='gray invalid/indoor; all metrics valid-only',benchmark='256 interpolated, not native new data'))
    write(out/(tag+'_selections.json'),dict(cases=selections,coverage=coverage))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--final',action='store_true')
    args=parser.parse_args();root=args.root.resolve();out=root/'visualizations';out.mkdir(exist_ok=True)
    priors(root,out);curves(root,out)
    for epoch in range(100,1001,100):
        files={v:root/'8x8'/v/'milestones'/f'epoch_{epoch:04d}'/'val.json' for v in SOURCES}
        name=f'val_epoch_{epoch:04d}'
        if all(p.exists() for p in files.values()):
            if not ready(out/name):
                compare_steps({v:json.loads(p.read_text()) for v,p in files.items()},out,name,f'Validation at matched epoch {epoch}')
            maps(root,out,'val',Path('milestones')/f'epoch_{epoch:04d}'/'val_predictions',name)
    if args.final:
        data={v:json.loads((root/'8x8'/v/'test_diagnostics.json').read_text()) for v in SOURCES}
        compare_steps(data,out,'test_euler_comparison','Final-epoch EMA: source-only test comparison')
        maps(root,out,'test',Path('test_diagnostics_predictions'),'test_final')
    images=sorted(out.glob('*.png'))
    text='# External source-distribution visualizations\n\nSource examples are not model predictions; curves include committed epochs only. Model comparisons are plotted after all five groups reach the same epoch.\n\n'
    text+='All results use seed 42; random realizations are not independent training seeds. Gray marks invalid/building pixels; displayed errors saturate above 30 dB.\n\n'
    for p in images:text+=f'## {p.stem}\n\n![{p.stem}]({p})\n\n[PDF]({p.with_suffix(".pdf")}) · [SVG]({p.with_suffix(".svg")})\n\n'
    write(out/'README.md',text)
    write(out/'figure_contract.json',dict(backend='python',destination='exploratory experiment report',dpi=300,
          primary_question='Does external source covariance affect correct detail recovery?',
          uncertainty='one training seed; where shown SD across four realizations',
          formats=['png','pdf','svg'],visual_review='Initial figures manually inspected; future automated exports require human review'))


if __name__=='__main__':main()
