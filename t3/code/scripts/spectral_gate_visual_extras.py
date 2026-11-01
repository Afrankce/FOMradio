"""Actual committed predictions: native Haar, mask-safe patch spectrum, gate maps."""
import json
import numpy as np


def native_and_spectral(file,variant,epoch):
    with np.load(file,allow_pickle=False) as sample:
        mask=sample['valid_mask'][0].astype(bool)
        y=sample['target'][0].astype(float)*300-300
        p=np.clip(sample['prediction'][0],0,1).astype(float)*300-300
    identity=dict(variant=variant,epoch=epoch,scene=file.stem,steps=8,realization=int(file.parent.name[5:]))
    wave=[];valid=mask.copy();a,b=p.copy(),y.copy()
    for level in (1,2,3):
        def transform(x):
            aa,bb,cc,dd=x[::2,::2],x[::2,1::2],x[1::2,::2],x[1::2,1::2]
            return (aa+bb+cc+dd)/2,[(aa-bb+cc-dd)/2,(aa+bb-cc-dd)/2,(aa-bb-cc+dd)/2]
        valid=valid[::2,::2]&valid[::2,1::2]&valid[1::2,::2]&valid[1::2,1::2]
        a,ad=transform(a);b,bd=transform(b)
        for direction,pa,ya in zip(('column','row','diagonal'),ad,bd):
            pp,yy=pa[valid],ya[valid]
            wave.append(dict(**identity,level=level,orientation=direction,count=len(pp),
                mse=float(np.mean((pp-yy)**2)) if len(pp) else None,
                signed_correlation=float(np.corrcoef(pp,yy)[0,1]) if len(pp)>1 and pp.std()>0 and yy.std()>0 else None))
    size=32;win=np.outer(np.hanning(size),np.hanning(size));correction=(win**2).mean()
    fy,fx=np.meshgrid(np.fft.fftfreq(size),np.fft.fftfreq(size),indexing='ij')
    rho=np.sqrt(fx**2+fy**2)
    bands={'low':(rho>0)&(rho<.125),'mid':(rho>=.125)&(rho<.25),'high':rho>=.25}
    values={k:[] for k in bands};possible=0;accepted=0
    for i in range(0,p.shape[0]-size+1,size):
        for j in range(0,p.shape[1]-size+1,size):
            possible+=1
            if not mask[i:i+size,j:j+size].all():continue
            accepted+=1
            pp=p[i:i+size,j:j+size];yy=y[i:i+size,j:j+size]
            fp=np.fft.fft2((pp-pp.mean())*win,norm='ortho')
            fy_=np.fft.fft2((yy-yy.mean())*win,norm='ortho')
            amplitude=(abs(fp)-abs(fy_))**2
            # Algebraically exact amplitude-weighted phase error; zero-energy safe.
            phase=np.maximum(2*(abs(fp)*abs(fy_)-(fp*np.conj(fy_)).real),0)
            for band,support in bands.items():
                values[band].append((float(amplitude[support].sum()/size**2/correction),
                                     float(phase[support].sum()/size**2/correction)))
    spec=[dict(**identity,band=band,patch_size=size,patches=accepted,possible_patches=possible,
               amplitude_error=float(np.mean([v[0] for v in rows])) if rows else None,
               phase_alignment_error=float(np.mean([v[1] for v in rows])) if rows else None)
          for band,rows in values.items()]
    return wave,spec


def render_extras(root,out,stages,histories):
    from scripts.visualize_spectral_gate import VARIANTS,COLORS,csv_out,line_plot,export,plt
    common=set.intersection(*(set(stages[v]) for v in VARIANTS))
    if common:
        epoch=max(common);wave=[];spec=[]
        for v in VARIANTS:
            for file in sorted((stages[v][epoch]/'N8').glob('seed_*/*.npz')):
                w,s=native_and_spectral(file,v,epoch);wave.extend(w);spec.extend(s)
        csv_out(out/f'native_haar_epoch_{epoch:04d}.csv',wave)
        csv_out(out/f'patch_spectral_epoch_{epoch:04d}.csv',spec)
        for metric,label in [('amplitude_error','Amplitude error'),('phase_alignment_error','Phase-alignment error')]:
            series=[]
            for v in VARIANTS:
                vals=[]
                for band in ('low','mid','high'):
                    # Average draws/patch estimates within scene before across scenes.
                    rows=[r for r in spec if r['variant']==v and r['band']==band and r[metric] is not None]
                    scenes={r['scene'] for r in rows}
                    vals.append(float(np.mean([np.mean([r[metric] for r in rows if r['scene']==s]) for s in scenes])) if scenes else np.nan)
                series.append((v,[1,2,3],vals,COLORS[v],'-'))
            line_plot(out,f'patch_{metric}_epoch_{epoch:04d}',f'Epoch {epoch}: valid 32-pixel patches; bands 1/2/3 = low/mid/high',
                      label+' (window-corrected dB²)',series,xlabel='Spatial-frequency band (1: low, 2: mid, 3: high)')
        # A single matched scene, all methods at exactly the same epoch/draw.
        files=sorted((stages['B'][epoch]/'N8/seed_0').glob('*.npz'))
        if files:
            file=files[0];fields=[]
            for v in VARIANTS:
                with np.load(stages[v][epoch]/'N8/seed_0'/file.name,allow_pickle=False) as a:
                    fields.append(np.clip(a['prediction'][0],0,1)*300-300)
                    target=a['target'][0]*300-300;mask=a['valid_mask'][0].astype(bool)
            fig,axes=plt.subplots(1,5,figsize=(15,3.8))
            fig.subplots_adjust(left=.02,right=.98,top=.80,bottom=.18,wspace=.15)
            cmap=plt.get_cmap('jet').copy();cmap.set_bad('#000000')
            for ax,field,label in zip(axes,[target,*fields],['GT',*VARIANTS]):
                ax.imshow(np.ma.array(field,mask=~mask),cmap=cmap,vmin=-300,vmax=0)
                ax.set_title(label);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f'{file.stem} | Epoch {epoch} | Online weights | N8, source draw 0',y=.96)
            fig.text(.5,.07,'Shared map scale: -300 to 0 dB; black: invalid/building. Fixed validation scene, not error-selected.',ha='center',fontsize=9)
            export(fig,out/f'comparison_epoch_{epoch:04d}')
    # Module-specific raw gradient norms, preserving zero-gradient diagnostics.
    gradients=[]
    for v in VARIANTS:
        end=histories[v][-1]['epoch'] if histories[v] else 0
        for file in (root/'8x8'/v/'loss_diagnostics').glob('window_*.json'):
            row=json.loads(file.read_text())
            if row['epoch_index']>=end:continue
            for group,stats in (row['parameter_gradients'] or {}).items():
                if group.startswith('gate_'):
                    gradients.append(dict(variant=v,epoch=row['epoch_index']+1,group=group,**stats))
    csv_out(out/'gate_gradients.csv',gradients)
    for v in ('R','A','AP'):
        series=[]
        for group,color in zip(('gate_radial','gate_amplitude_head','gate_phase_head','gate_condition_alignment','gate_descriptor_network'),
                               ('#777777','#3975A4','#8B6FA5','#6B8C67','#BC6C45')):
            rows=sorted([r for r in gradients if r['variant']==v and r['group']==group],key=lambda r:r['epoch'])
            if rows:series.append((group.replace('gate_',''),[r['epoch'] for r in rows],[r['total_norm'] for r in rows],color,'-'))
        if series:line_plot(out,v+'_gate_gradient_norms',v+': total raw gradient before Adam','Gradient norm',series)
        # Milestones have deterministic actual-flow gate probes; don't relabel smoke as training.
        probes=sorted((root/'8x8'/v/'milestones').glob('epoch_*/gate_probe.json'))
        if not probes:continue
        directory=probes[-1].parent
        file=directory/'gate_encoder_stages.0.spectral_gate_t4.npz'
        if not file.exists():continue
        with np.load(file,allow_pickle=False) as arr:
            order=np.argsort(arr['ky'])
            amp=arr['amplitude'].mean(0)[order];phase=arr['phase'][0][order];spatial=arr['spatial_delta_rms']
        fig,axes=plt.subplots(1,3,figsize=(10.5,3.8))
        fig.subplots_adjust(left=.04,right=.96,top=.75,bottom=.23,wspace=.3)
        for ax,field,title,cmap,lo,hi in zip(axes,[amp,phase,spatial],
                ['Amplitude (channel mean)','Phase (channel 0, rad)','Spectral-output change RMS'],
                ['coolwarm','twilight','magma'],[.5,-np.pi/4,0],[1.5,np.pi/4,float(spatial.max()) or 1.]):
            ax.imshow(field,cmap=cmap,vmin=lo,vmax=hi,aspect='auto');ax.set_title(title,fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'{v} | {directory.name} | First encoder stage | Actual flow t=0.5',y=.97)
        fig.text(.5,.07,'Amplitude: 0.5–1.5; phase: ±pi/4. RMS: 0–'+f'{float(spatial.max()):.4g}'+' (latent units, padded grid).',ha='center',fontsize=8)
        export(fig,out/f'gate_{v}_{directory.name}')
