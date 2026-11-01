"""Completed-source test report; all test scenes contribute to quantitative plots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import visualize_source_suite as v
import matplotlib as mpl
mpl.rcParams.update({'font.family':'sans-serif','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})


def paired(data,out):
    scenes=sorted({r['scene_id'] for r in data['W']['records']})
    fig,axes=v.frame('Paired test effects: MS versus completed baselines',
        'N=8; 160 scenes; four realizations averaged within scene; negative difference favors MS',cols=2)
    numeric=[]
    for ax,key,title in zip(axes,('db_rmse','fine_mse'),('a  Pointwise trade-off','b  Detail trade-off')):
        def values(name):
            return np.array([np.mean([r[key] for r in data[name]['records'] if r['scene_id']==s and r['steps']==8]) for s in scenes])
        candidate=values('MS')
        for i,baseline in enumerate(('W','S2','S8')):
            difference=candidate-values(baseline)
            offset=np.linspace(-.17,.17,len(scenes))
            ax.scatter(difference,i+offset,color=v.COLORS[v.SOURCES.index(baseline)],s=9,alpha=.45,rasterized=True)
            rng=np.random.default_rng(20260909)
            ci=np.quantile(difference[rng.integers(0,len(scenes),(10000,len(scenes)))].mean(1),[.025,.975])
            mean=difference.mean()
            ax.errorbar(mean,i,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt='D',color='black',ms=5,capsize=5,zorder=4)
            numeric.append(dict(metric=key,baseline=baseline,scene_ids=scenes,paired_differences=difference.tolist(),
                mean=float(mean),ci95=ci.tolist(),wins=int((difference<0).sum())))
        ax.axvline(0,color='#666666',lw=.8,ls='--');ax.set_yticks(range(3),['MS − W','MS − S2','MS − S8'])
        ax.set(title=title,xlabel='Difference (dB)' if key=='db_rmse' else 'Difference (dB²)',ylim=(-.5,2.5))
    v.save(fig,axes,out/'paired_test_effects',dict(values=numeric,
        uncertainty='Black diamond: paired scene mean; 95% descriptive bootstrap CI, 10000 resamples; not training-seed uncertainty',
        no_p_values=True,test_status='exploratory previously observed holdout'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();out=root/'completed_test_4groups';out.mkdir(exist_ok=True)
    audit=json.loads((out/'test_audit.json').read_text());assert audit['status']=='passed'
    colors=dict(zip(v.SOURCES,v.COLORS));v.SOURCES=('W','S2','S8','MS');v.COLORS=tuple(colors[n] for n in v.SOURCES)
    data={n:json.loads((root/'8x8'/n/'test_diagnostics.json').read_text()) for n in v.SOURCES}
    for n in v.SOURCES:assert hashlib.sha256((root/'8x8'/n/'test_diagnostics.json').read_bytes()).hexdigest()==audit['sources'][n]['test_json_sha256']
    v.compare_steps(data,out,'test_euler_comparison','Completed models: final-epoch EMA on 160 test scenes')
    paired(data,out)
    v.maps(root,out,'test',Path('test_diagnostics_predictions'),'test_N8')
    v.maps(root,out,'test',Path('test_diagnostics_predictions'),'test_N2',steps=2,scenes=['u799'])
    rows=[]
    for name in v.SOURCES:
        for n in (2,4,8,12):
            records=[r for r in data[name]['records'] if r['steps']==n]
            aggregate=[r for r in data[name]['aggregated'] if r['steps']==n]
            row=dict(source=name,steps=n,scenes=160,realizations=4,
                     **{k:float(np.mean([r[k] for r in records])) for k in ('db_rmse','db_bias','raw_db_rmse','fine_mse','horizontal_lag1_gradient_mse')},
                     ssim=float(np.mean([r['ssim'] for r in aggregate])))
            rows.append(row)
    with (out/'test_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report='# Four completed groups: final test and visualizations\n\n'
    report+='W, S2, S8, and MS: epoch-1000 EMA; 160 test scenes x four realizations x N=2/4/8/12. S32 is incomplete and excluded. Existing full-test outputs are reused and audited without repeated inference.\n\n'
    report+='RMSE and detail MSE average per-prediction metrics over scenes/realizations, not metrics of an ensemble-mean map. SSIM averages valid-window summaries across realizations.\n\n'
    report+='| Source | N | RMSE (dB) ↓ | SSIM ↑ | Fine MSE (dB²) ↓ | Bias (dB) |\n|---|---:|---:|---:|---:|---:|\n'
    for r in rows:report+=f"| {r['source']} | {r['steps']} | {r['db_rmse']:.4f} | {r['ssim']:.4f} | {r['fine_mse']:.4f} | {r['db_bias']:.4f} |\n"
    report+='\nOne training seed (42); four generations are not independent training replicates. This test set was explored previously, so these results are not fresh confirmatory evidence. The preregistered primary endpoint remains N=8 and is not reselected from this table.\n\n'
    report+='Displayed error colors saturate at 30 dB; statistics are not truncated there. Gray marks invalid/building regions. Crops use a common target-quantile range to expose detail; full maps use -300 to 0 dB.\n\n'
    report+='Step-curve error bars are SD across scene means from four realizations. Each colored paired point is one scene; the black diamond is the mean difference and the horizontal bar is a descriptive 95% interval from 10,000 scene-bootstrap draws, not training-seed uncertainty. Gray building outlines come from known label masks and are not model predictions.\n\n'
    report+='[Numeric CSV](test_summary.csv) · [Test audit](test_audit.json) · [Figure notes](figure_contract.json)\n\n'
    for image in [out/'test_euler_comparison.png',out/'paired_test_effects.png',*sorted(out.glob('test_N*.png'))]:
        report+=f'## {image.stem}\n\n![{image.stem}]({image})\n\n[PDF]({image.with_suffix(".pdf")}) · [SVG]({image.with_suffix(".svg")})\n\n'
    v.write(out/'README.md',report)
    v.write(out/'render_provenance.json',dict(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(v.__file__).read_bytes()).hexdigest(),
        inference='No new inference; exact existing final test reused',all_quantitative_scenes=160,selected_visual_scenes='u799 plus geometry coverage quartiles',
        training_unchanged=True,formats=['png300dpi','pdf','svg'],static_warnings='Report dimensions/PNG300dpi intentional; not journal submission'))


if __name__=='__main__':main()
