"""Render stored, immutable Matérn experiment evidence (no new inference)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

VARIANTS = ['B2', 'M1', 'M2', 'M3']
COLORS = ['#424b54', '#4a83ad', '#cf9557', '#9673ae']
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .8})


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--qa-scripts', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'visualizations'
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    exports = []
    sources = {}

    def source(path):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def save(fig, name, axes, metadata):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, axes=list(np.asarray(axes).flat),
            json_out=out / f'{name}.alignment.json', strict=True)
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'{name}.{suffix}', dpi=300, facecolor='white')
        write_json(out / f'{name}.json', metadata)
        plt.close(fig)
        exports.append(name)

    def heading(fig, title, subtitle):
        fig.suptitle(title, y=.98, fontsize=14, fontweight='bold')
        fig.text(.5, .89 if fig.get_figheight() < 5.5 else .922, subtitle, ha='center', fontsize=10)

    # All 1,000 epochs; one training seed, no fabricated uncertainty interval.
    histories = {v: list(csv.DictReader(source(root/'8x8'/v/'metrics.csv').open())) for v in VARIANTS}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    fig.subplots_adjust(left=.065, right=.98, bottom=.19, top=.78, wspace=.32)
    for v, color in zip(VARIANTS, COLORS):
        rows = histories[v]
        x = [int(r['epoch']) for r in rows]
        for ax, key in zip(axes, ['train_loss', 'val_db_rmse', 'val_db_rmse']):
            visible = [r for r in rows if int(r['epoch']) >= 800] if ax is axes[2] else rows
            ax.plot([int(r['epoch']) for r in visible], [float(r[key]) for r in visible], color=color, label=v, lw=1.5)
    axes[0].set_yscale('log')
    axes[2].set_xlim(800, 1000)
    late = [float(r['val_db_rmse']) for rows in histories.values() for r in rows if int(r['epoch']) >= 800]
    axes[2].set_ylim(min(late)-.12, max(late)+.12)
    for ax, title, ylabel in zip(axes, ['a  Optimization', 'b  Validation', 'c  Final 200 epochs'],
                               ['Masked velocity MSE', 'Validation RMSE (dB)', 'Validation RMSE (dB)']):
        ax.set(title=title, xlabel='Epoch', ylabel=ylabel)
        ax.grid(axis='y', alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4)
    heading(fig, 'Training convergence: matched 8 x 8 ablations', 'All epochs | training seed 42 | validation: N=2, CFG=1, fixed realization')
    save(fig, 'training_curves', axes, dict(n_training_seeds=1, epoch_range=[1,1000], smoothing='none'))

    diagnostics = {v: json.loads(source(root/'8x8'/v/'test_diagnostics.json').read_text()) for v in VARIANTS}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    fig.subplots_adjust(left=.065, right=.98, bottom=.19, top=.78, wspace=.32)
    numerical = []
    for v, color in zip(VARIANTS, COLORS):
        for ax, key in zip(axes, ['db_rmse', 'ssim', 'fine_mse']):
            groups = []
            for n in [2,4,8,12]:
                if key != 'fine_mse':
                    values = [r[key] for r in diagnostics[v]['aggregated'] if r['steps']==n]
                else:
                    values = [np.mean([r[key] for r in diagnostics[v]['records'] if r['steps']==n and r['realization']==seed]) for seed in range(4)]
                groups.append(values)
                numerical.append(dict(variant=v, steps=n, metric=key, mean=float(np.mean(values)), sd=float(np.std(values,ddof=1))))
            array = np.array(groups)
            ax.errorbar([2,4,8,12], array.mean(1), yerr=array.std(1,ddof=1), marker='o', color=color,
                        lw=1.5, markersize=4, capsize=3, label=v)
    for ax, title, ylabel in zip(axes, ['a  Pointwise fidelity', 'b  Structural similarity', 'c  Fine-detail fidelity'],
                                ['Test RMSE (dB)', 'Test SSIM', 'Haar detail MSE (dB²)']):
        ax.set(title=title, xlabel='Euler steps N', ylabel=ylabel, xticks=[2,4,8,12])
        ax.grid(axis='y', alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4)
    heading(fig, 'More integration steps do not reduce reconstruction error', '160 test scenes | same best EMA | mean +/- SD of 4 realizations, not training seeds')
    save(fig, 'euler_step_comparison', axes, dict(source_data=numerical, excluded_scenes=0,
        fine_metric='Scene-mean fine MSE; complete-support Haar details after shared 2x2 benchmark averaging'))

    # Scene selection uses geometry only, with u799 added as a user-requested case.
    config = json.loads(source(root/'8x8/B2/config.json').read_text())
    dataset = Path(config['dataset_root'])
    records = [json.loads(line) for line in source(root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl').read_text().splitlines()]
    records = {r['scene_id']: r for r in records if r['split']=='test'}
    coverage = []
    for scene, row in records.items():
        native = np.load(dataset/row['radiomap_path'])
        coverage.append((float(np.mean(native==1000)), scene))
    coverage.sort()
    chosen = [('u799','Previously discussed case')]
    for quantile in [.25,.5,.75]:
        frac, scene = coverage[round(quantile*(len(coverage)-1))]
        if scene not in [x[0] for x in chosen]:
            chosen.append((scene,f'Building-coverage quantile {quantile:.2f}'))
    height_max = max(float(np.load(dataset/records[s]['height_path']).max()) for s,_ in chosen)
    cmap = plt.colormaps['viridis'].copy()
    cmap.set_bad('#b0b0b0')
    errors = plt.colormaps['magma'].copy()
    errors.set_bad('#b0b0b0')
    scene_metadata = []
    for scene, selection in chosen:
        files = {v: source(root/'8x8'/v/'test_predictions/seed_0'/f'{scene}.npz') for v in VARIANTS}
        stored = {v: np.load(path) for v,path in files.items()}
        gt = stored['B2']['target'][0]*300-300
        mask = stored['B2']['valid_mask'][0]
        for v in VARIANTS:
            assert np.array_equal(stored[v]['valid_mask'][0],mask)
            assert np.array_equal(stored[v]['target'][0],stored['B2']['target'][0])
        predictions = {v: np.clip(stored[v]['prediction'][0],0,1)*300-300 for v in VARIANTS}
        height = np.load(dataset/records[scene]['height_path'])
        # Select the height edge nearest the central transmitter; no error-based crop.
        gy,gx = np.gradient(height)
        edges = np.argwhere(np.hypot(gx,gy)>0)
        center = edges[np.argmin(((edges-np.array([127,127]))**2).sum(1))] if len(edges) else np.array([127,127])
        top,left = np.clip(center-32,0,192).astype(int)
        crop = (slice(top,top+64),slice(left,left+64))
        info = dict(scene=scene, selection=selection, building_fraction=dict((s,f) for f,s in coverage)[scene],
                    realization=0, steps=2, cfg=1, crop_rc=[int(top),int(left),64,64],
                    masks='All invalid/indoor cells shown gray; no imputation', grid='256 interpolated benchmark pixels; native radio is 128²',
                    height_sha256=hashlib.sha256((dataset/records[scene]['height_path']).read_bytes()).hexdigest(),
                    per_method={v:dict(rmse_db=float(np.sqrt(np.mean((predictions[v][mask]-gt[mask])**2))),
                                      error_above_30_fraction=float(np.mean(np.abs(predictions[v][mask]-gt[mask])>30))) for v in VARIANTS})
        scene_metadata.append(info)
        fig, axes = plt.subplots(2,5,figsize=(14,7))
        fig.subplots_adjust(left=.035,right=.965,bottom=.20,top=.84,wspace=.12,hspace=.25)
        values = [gt]+[predictions[v] for v in VARIANTS]
        for ax,title,value in zip(axes[0],['Ground truth']+VARIANTS,values):
            im = ax.imshow(np.ma.array(value,mask=~mask),cmap=cmap,vmin=-300,vmax=0,interpolation='nearest')
            ax.add_patch(Rectangle((left-.5,top-.5),64,64,edgecolor='white',fill=False,lw=1))
            ax.set_title(title)
        hm=axes[1,0].imshow(height,cmap='gray',vmin=0,vmax=height_max,interpolation='nearest')
        axes[1,0].plot(127,127,marker='+',color='#e05b55',markersize=9)
        axes[1,0].set_title('Height + Tx')
        for ax,v in zip(axes[1,1:],VARIANTS):
            err=np.abs(predictions[v]-gt)
            em=ax.imshow(np.ma.array(err,mask=~mask),cmap=errors,vmin=0,vmax=30,interpolation='nearest')
            ax.set_title(f'{v} error | RMSE {info["per_method"][v]["rmse_db"]:.2f} dB',fontsize=9)
        for ax in axes.flat:
            ax.set_xticks([]);ax.set_yticks([])
        for image,position,label,extend in [(hm,[.05,.09,.16,.018],'Height (dataset units)','neither'),
                         (im,[.29,.09,.26,.018],'Radiomap (dB)','neither'),(em,[.67,.09,.26,.018],'Absolute error (dB); >30 saturated','max')]:
            cb=fig.colorbar(image,cax=fig.add_axes(position),orientation='horizontal',extend=extend)
            cb.set_label(label,fontsize=9)
        heading(fig,f'{scene} | Ground truth, predictions and errors',f'8 x 8 | N=2 | realization 0 | {selection} | white box: common crop')
        save(fig,f'{scene}_comparison',axes,info)

        fig,axes=plt.subplots(2,5,figsize=(14,7))
        fig.subplots_adjust(left=.035,right=.965,bottom=.20,top=.84,wspace=.12,hspace=.25)
        limits=np.percentile(gt[crop][mask[crop]],[1,99])
        for ax,title,value in zip(axes[0],['Ground truth']+VARIANTS,values):
            im=ax.imshow(np.ma.array(value[crop],mask=~mask[crop]),cmap=cmap,vmin=limits[0],vmax=limits[1],interpolation='nearest')
            ax.set_title(title)
        hm=axes[1,0].imshow(height[crop],cmap='gray',vmin=0,vmax=height_max,interpolation='nearest')
        axes[1,0].set_title('Height crop')
        for ax,v in zip(axes[1,1:],VARIANTS):
            err=np.abs(predictions[v][crop]-gt[crop])
            em=ax.imshow(np.ma.array(err,mask=~mask[crop]),cmap=errors,vmin=0,vmax=30,interpolation='nearest')
            rmse=np.sqrt(np.mean(err[mask[crop]]**2))
            ax.set_title(f'{v} crop error | {rmse:.2f} dB',fontsize=9)
        for ax in axes.flat:
            ax.set_xticks([]);ax.set_yticks([])
        for image,position,label,extend in [(hm,[.05,.09,.16,.018],'Height (dataset units)','neither'),
                         (im,[.29,.09,.26,.018],'Radiomap (dB); shared crop contrast','both'),(em,[.67,.09,.26,.018],'Absolute error (dB); >30 saturated','max')]:
            cb=fig.colorbar(image,cax=fig.add_axes(position),orientation='horizontal',extend=extend)
            cb.set_label(label,fontsize=9)
        heading(fig,f'{scene} | Local structure near a building edge',f'Same 64 x 64 crop | geometry-only selection | nearest-neighbor display | GT 1-99% color range')
        save(fig,f'{scene}_crop',axes,{**info,'radio_color_limits_db':limits.tolist(),
            'display_clipping_fraction': {v:float(np.mean((predictions[v][crop][mask[crop]]<limits[0]) | (predictions[v][crop][mask[crop]]>limits[1]))) for v in VARIANTS}})

    write_json(out/'source_manifest.json',sources)
    write_json(out/'scene_selection.json',dict(all_test_coverage=coverage,selected=scene_metadata))
    write_json(out/'figure_contract.json',dict(backend='python',purpose='Exploratory result inspection, not submission',
        claim='Matched noise/feedback ablations preserve global structure but differ subtly in local fidelity; more steps trade metrics',
        archetypes=['quantitative grid','image plate + quant'],formats=['PNG 300dpi','PDF','SVG'],
        integrity='No source arrays modified; crops/limits recorded; seed 0 examples, all seeds in aggregate step plots',
        physical_sizes_inches={'curves':[13.2,4.7],'plates':[14,7]},minimum_font_pt=9))
    text=['# Matérn experiment visualizations','', 'Generated from saved results without retraining or rerunning inference.',
          'Curves cover all training epochs and test scenes; examples fix realization=0 and do not represent a four-draw average.',
          'B2 uses white noise at each level; M1 uses fixed cross-level Matérn correlation length; M2 uses multiscale Matérn; M3 adds feedback.',
          'Gray denotes invalid/building-interior regions. Full maps share -300 to 0 dB; crops share the corresponding ground-truth 1st-to-99th percentile range across all four models.',
          'Error colors span 0 to 30 dB and saturate above 30 dB; this does not imply all errors are at most 30 dB.',
          'Original radiomaps are 128 squared; 256 squared in the figures is the interpolated benchmark grid.', '']
    for name in exports:
        text += [f'## {name}','',f'![{name}]({(out/(name+".png")).resolve()})','',f'[PDF]({(out/(name+".pdf")).resolve()}) · [SVG]({(out/(name+".svg")).resolve()})','']
    (out/'README.md').write_text('\n'.join(text))
    print(json.dumps(dict(output=str(out),figures=exports),ensure_ascii=False))


if __name__=='__main__':
    main()
