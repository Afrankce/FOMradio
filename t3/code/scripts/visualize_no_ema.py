"""Reuse audited diagnostic plots and add explicitly labeled EMA history overlays."""
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.visualize_conditional_operator import run,line_plot,COLORS,VARIANTS
from scripts.monitor_matern_suite import committed_history,atomic
from scripts.analyze_ema_comparison import analyze


def render(root):
    run(root)
    out=root/'visualizations'
    reference=Path(json.loads((root/'execution_scope.json').read_text())['reference'])
    for v in VARIANTS:
        current,_=committed_history(root/'8x8'/v)
        old,_=committed_history(reference/'8x8'/v)
        if not current:continue
        end=current[-1]['epoch'];old=[r for r in old if r['epoch']<=end]
        line_plot(out,'ema_overlay_'+v,v+': online vs archived EMA (N=2)','Global valid-pixel RMSE (dB)',[
            ('No EMA',[r['epoch'] for r in current],[r['val_db_rmse'] for r in current],COLORS[v],'-'),
            ('Archived EMA',[r['epoch'] for r in old],[r['val_db_rmse'] for r in old],'#777777','--')])
    index=out/'README.md'
    text=index.read_text().replace('# Conditional operator live diagnostics','# No-EMA: online optimizer weights')
    text+='\n\nAll new-run predictions use online weights without averaging. Legacy ema.pt / ema fields contain identical online weights, not EMA.\n'
    text+='\nHistory overlays compare separate matched-seed runs; check trajectory audits before attributing differences solely to EMA. Endpoint paired evidence: [summary](../ema_comparison/summary.md).\n'
    text+='\n'.join(f'\n![{v} online vs EMA](ema_overlay_{v}.png)\n' for v in VARIANTS if (out/f'ema_overlay_{v}.png').exists())
    atomic(index,text);analyze(root)
    from scripts.visualize_ema_endpoints import render as paired_maps
    paired_maps(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    render(p.parse_args().root.resolve())
