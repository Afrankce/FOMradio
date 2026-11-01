"""Structural adaptation of the existing test map template; four condition variants."""
import json
from pathlib import Path
import numpy as np
from matplotlib.patches import Rectangle
from scripts.visualize_source_suite import plt,save,write,ready
SOURCES=('CASA','ADD','FILM','NSOP')

def maps(root,out,split,subdirectory,tag,steps=8,scenes=None):
    manifest=[json.loads(x) for x in (root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl').read_text().splitlines()]
    records={r['scene_id']:r for r in manifest if r['split']==split}
    config=json.loads((root/'8x8/CASA/config.json').read_text());dataset=Path(config['dataset_root'])
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
        gt=stored['CASA']['target'][0]*300-300;mask=stored['CASA']['valid_mask'][0]
        for v in SOURCES:
            assert np.array_equal(stored[v]['target'],stored['CASA']['target'])
            assert np.array_equal(stored[v]['valid_mask'],stored['CASA']['valid_mask'])
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
