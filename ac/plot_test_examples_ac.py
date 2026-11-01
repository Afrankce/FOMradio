"""Plot the three preselected array-test scenes (seed 42, fixed draw 0)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT=Path(__file__).resolve().parent


def main():
    spec=json.loads((ROOT/'test_run_spec.json').read_text())
    out=ROOT/spec['test_output_root'];a=out/'A_seed42';c=out/'C_seed42'
    selection=json.loads((a/'selection.json').read_text());other=json.loads((c/'selection.json').read_text())
    if selection!=other or len(selection['scene_ids'])!=3:raise RuntimeError('selection mismatch')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','DejaVu Serif'],
                         'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(3,5,figsize=(12,7.8),layout='constrained')
    titles=('Ground truth','A','A absolute error','C','C absolute error')
    for ax,title in zip(axes[0],titles):ax.set_title(title)
    cmap=plt.colormaps['turbo'].copy();cmap.set_bad('#444444')
    errmap=plt.colormaps['magma'].copy();errmap.set_bad('#444444')
    for row,scene in enumerate(selection['scene_ids']):
        with np.load(a/'selected_maps'/f'{scene}_draw0.npz',allow_pickle=False) as za:
            ya,pa,mask=za['ground_truth_db'],za['prediction_db'],za['valid']
        with np.load(c/'selected_maps'/f'{scene}_draw0.npz',allow_pickle=False) as zc:
            yc,pc,maskc=zc['ground_truth_db'],zc['prediction_db'],zc['valid']
        if not np.array_equal(ya,yc) or not np.array_equal(mask,maskc):raise RuntimeError('paired target mismatch')
        maps=(ya,pa,np.abs(pa-ya),pc,np.abs(pc-ya))
        for col,(ax,value) in enumerate(zip(axes[row],maps)):
            ax.imshow(np.ma.masked_where(~mask,value),cmap=errmap if col in (2,4) else cmap,
                      vmin=0 if col in (2,4) else -300,vmax=100 if col in (2,4) else 0,
                      origin='upper',interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(scene,rotation=90)
    fig.colorbar(ScalarMappable(norm=Normalize(-300,0),cmap=cmap),ax=axes[:,[0,1,3]],
                 location='bottom',fraction=.025,pad=.04,label='Power (dB)')
    fig.colorbar(ScalarMappable(norm=Normalize(0,100),cmap=errmap),ax=axes[:,[2,4]],
                 location='bottom',fraction=.025,pad=.04,label='Absolute error (dB)')
    for ext in ('pdf','png'):fig.savefig(out/f'array_examples_seed42_draw0.{ext}',dpi=200)
    plt.close(fig)
    print(json.dumps({'scenes':selection['scene_ids'],'seed':42,'draw':0,
                      'files':[str(out/f'array_examples_seed42_draw0.{x}') for x in ('pdf','png')]}))


if __name__=='__main__':main()
