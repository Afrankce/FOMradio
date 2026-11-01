"""Fixed validation scenes, paired weights from exactly the same checkpoint."""
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
from scripts.visualize_conditional_operator import plt,export,atomic,VARIANTS


def render(root):
    reference=Path(json.loads((root/'execution_scope.json').read_text())['reference'])
    out=root/'ema_comparison';out.mkdir(exist_ok=True)
    for v in VARIANTS:
        online_dir=root/'8x8'/v/'reference_online'
        if not (online_dir/'val_diagnostics.json').exists():continue
        old_dir=reference/'8x8'/v/'val_diagnostics_predictions/N8/seed_0'
        for p in sorted(old_dir.glob('*.npz'))[:4]:
            dst=out/f'paired_{v}_{p.stem}'
            if dst.with_suffix('.qa.json').exists():continue
            old=np.load(p,allow_pickle=False)
            new=np.load(online_dir/'val_diagnostics_predictions/N8/seed_0'/p.name,allow_pickle=False)
            if not np.array_equal(old['target'],new['target']) or not np.array_equal(old['valid_mask'],new['valid_mask']):
                raise ValueError('Mismatched paired targets or masks')
            mask=old['valid_mask'][0].astype(bool);y=old['target'][0]*300-300
            a=np.clip(old['prediction'][0],0,1)*300-300;b=np.clip(new['prediction'][0],0,1)*300-300
            fig,axes=plt.subplots(2,3,figsize=(10.8,7.4))
            fig.subplots_adjust(left=.04,right=.96,bottom=.15,top=.88,wspace=.2,hspace=.23)
            fields=[y,a,b,b-a,abs(a-y),abs(b-y)]
            titles=['Ground truth','EMA prediction','Online prediction','Online minus EMA','EMA absolute error','Online absolute error']
            for ax,field,title,cmap,lo,hi in zip(axes.flat,fields,titles,
                    ['viridis']*3+['RdBu_r','magma','magma'],[-300]*3+[-40,0,0],[0]*3+[40,100,100]):
                cm=plt.get_cmap(cmap).copy();cm.set_bad('#b5b5b5')
                ax.imshow(np.ma.array(field,mask=~mask),cmap=cm,vmin=lo,vmax=hi)
                ax.set_title(title+' (dB)',fontsize=10);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f'{v} | {p.stem} | Same epoch-1000 checkpoint | N=8, source draw 0',fontsize=11,y=.96)
            fig.text(.5,.075,'Shared scales: maps [-300, 0]; difference [-40, 40]; errors [0, 100] dB. Gray: invalid.',ha='center',fontsize=8)
            export(fig,dst)
    images=sorted(p.name for p in out.glob('paired_*.png'))
    atomic(out/'visualizations.md','# Paired endpoint maps\n\nFixed first four lexicographic validation scenes; no error-based selection. Same training checkpoint, source draw and valid mask.\n\n'+
           '\n\n'.join(f'![{p}]({p})' for p in images))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    render(p.parse_args().root.resolve())
