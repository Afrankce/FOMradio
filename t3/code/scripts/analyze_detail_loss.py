"""Read-only scientific interpretation of committed loss and validation diagnostics."""
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
import statistics
from scripts.monitor_matern_suite import atomic


def read(path):
    if not path.exists():return []
    with path.open() as stream:return list(csv.DictReader(stream))


def mean(values):
    values=[float(v) for v in values if v not in ('',None)]
    return statistics.mean(values) if values else None


def analyze(root):
    out=root/'visualizations';out.mkdir(exist_ok=True)
    lines=['# Interim loss, gradient, and propagation-structure analysis','',
           'This report uses only committed epoch records. One training seed does not support cross-seed significance claims.',
           'L1 is the gradient-error experiment label, not the L1 norm; L2 is the Haar auxiliary term. Both weights are 0.1, but parameter-gradient strength was not matched.','']
    rows=read(out/'parameter_gradients.csv')
    if rows:
        latest=max(int(r['epoch']) for r in rows)
        selected=[r for r in rows if int(r['epoch'])>=max(1,latest-99)]
        lines+=['## Parameter-gradient diagnostics over the latest 100 epochs','',
                '| Variant | Module | Measurements | Mean weighted auxiliary/main norm ratio | Mean cosine | Negative-cosine fraction |',
                '|---|---|---:|---:|---:|---:|']
        groups=defaultdict(list)
        for r in selected:groups[(r['variant'],r['group'])].append(r)
        for (v,g),rr in sorted(groups.items()):
            ratio=mean([r['weighted_ratio'] for r in rr]);cos=mean([r['cosine'] for r in rr])
            valid=[float(r['cosine']) for r in rr if r['cosine']!='']
            conflict=sum(x<0 for x in valid)/len(valid) if valid else None
            fmt=lambda x:f'{x:.4f}' if x is not None else 'NA'
            lines.append(f'| {v} | {g} | {len(rr)} | {fmt(ratio)} | {fmt(cos)} | {fmt(conflict)} |')
        lines+=['','The norm ratio describes the auxiliary objective relative to the original parameter gradient, not its share of the actual Adam update.',
                'A negative cosine indicates conflicting gradient directions on a batch, not by itself a harmful auxiliary term; interpret it with later validation error.','']
    files=sorted(out.glob('haar_orientation_epoch_*.csv'))
    if files:
        f=files[-1];epoch=int(f.stem.rsplit('_',1)[1]);wave=read(f)
        region=read(out/f'regional_epoch_{epoch:04d}.csv')
        lines+=[f'## Epoch {epoch}: Haar coefficients and propagation maps','',
            'Haar describes spatial variation in dB maps. Column, row, and diagonal differences are not respectively direct, reflected, and diffracted paths.',
            'Only N=8 is summarized below. Noise draws are averaged within each scene, then scenes are weighted equally.','',
            '| Variant | Scale | Array-difference orientation | Scenes | Coefficient MSE | Signed-coefficient correlation |',
            '|---|---:|---|---:|---:|---:|']
        cells=defaultdict(lambda:defaultdict(list))
        for r in wave:
            if int(r['steps'])==8:cells[(r['variant'],r['level'],r['orientation'])][r['scene']].append(r)
        for (v,s,o),scenes in sorted(cells.items()):
            mse=mean([mean([r['mse'] for r in rr]) for rr in scenes.values()])
            corr=mean([mean([r['signed_coefficient_correlation'] for r in rr]) for rr in scenes.values()])
            lines.append(f'| {v} | {s} | {o} | {len(scenes)} | {mse:.4f} | '+(f'{corr:.4f}' if corr is not None else 'NA')+' |')
        lines+=['','| Variant | Region | Scenes | Scene-mean RMSE (dB) | Scene-mean bias (dB) |',
                '|---|---|---:|---:|---:|']
        cells=defaultdict(lambda:defaultdict(list))
        for r in region:
            if int(r['steps'])==8 and r['rmse']!='':cells[(r['variant'],r['region'])][r['scene']].append(r)
        for (v,g),scenes in sorted(cells.items()):
            rmse=mean([mean([r['rmse'] for r in rr]) for rr in scenes.values()])
            bias=mean([mean([r['bias'] for r in rr]) for rr in scenes.values()])
            lines.append(f'| {v} | {g} | {len(scenes)} | {rmse:.4f} | {bias:.4f} |')
        lines+=['','Building regions are defined by height>0. Distance thresholds are pixels on the current 256 grid and are not converted to meters.',
                'Near-building means at most 8 px and far means above 16 px; the transition region remains in global metrics but not these two groups.',
                'Haar is computed only on fully valid support; excluded boundaries receive no wavelet supervision.',
                'Lower coefficient MSE, higher correlation, and improved near-building error together would support recovery of environment-related spatial structure.',
                'Even joint improvement cannot identify reflection paths from power maps alone or establish Maxwell-equation consistency.']
        if not any(r['variant']=='L0' for r in wave):
            lines+=['','This early preview has no same-epoch L0 map, so no local benefit relative to L0 is claimed; alignment begins at the 100-epoch checkpoint.']
    else:lines+=['','Awaiting the first matched-epoch validation maps; no conclusion on electromagnetic propagation structure is available.']
    atomic(out/'analysis.md','\n'.join(lines)+'\n')
    atomic(out/'analysis_provenance.json',dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [out/'parameter_gradients.csv',*files[-1:],*(sorted(out.glob('regional_epoch_*.csv'))[-1:])]
                 if p.exists()},
        status='descriptive ongoing analysis; no training or checkpoint mutation'))
