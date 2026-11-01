"""Durable 30-second server heartbeat and committed-epoch analysis."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.monitor_matern_suite import atomic,utc,process_matches,committed_history,summarize,table

_render_job=None
_render_signature=None
_render_failed=False


def poll(root):
    global _render_job,_render_signature,_render_failed
    out=root/'monitoring';out.mkdir(exist_ok=True)
    state_path=root/'suite_status.json'
    if not state_path.exists():return False
    state=json.loads(state_path.read_text());terminal=state['status'] in ('complete','failed')
    alerts=[];latest={};signature=[]
    if not terminal and not process_matches(state.get('pid'),root):alerts.append('supervisor_missing')
    for v in ('CASA','ADD','FILM','NSOP'):
        directory=root/'8x8'/v;job=state['jobs'].get(v,{})
        if job.get('phase')=='failed':alerts.append(v+':job_failed')
        if job.get('phase') in ('training','smoke','evaluate_val','evaluate_test') and not terminal:
            if not process_matches(job.get('pid'),root):alerts.append(v+':process_missing')
            log=directory/(job['phase']+'.log')
            if log.exists() and time.time()-log.stat().st_mtime>1800:alerts.append(v+':progress_stale_30min')
        rows,_=committed_history(directory)
        if rows:
            latest[v]=summarize(rows,rows[-1]['epoch']);signature.append((v,rows[-1]['epoch']//20))
            for epoch in range(100,rows[-1]['epoch']+1,100):
                path=out/'statistics'/f'{v}_epoch_{epoch:04d}.json'
                if not path.exists():
                    records=[json.loads(p.read_text()) for p in (directory/'loss_diagnostics').glob('window_*.json')]
                    records=[r for r in records if epoch-100<=r['epoch_index']<epoch]
                    summary=summarize(rows,epoch)
                    summary['loss_diagnostics']=dict(windows=len(records),
                        component_batch_mean={key:statistics.mean(r['losses'][key] for r in records)
                                              for key in ('fm','gradient','haar1','haar2')},
                        skipped_optimizer_windows=sum(not r['optimizer_ran'] for r in records))
                    grouped={}
                    for r in records:
                        for group,g in (r['parameter_gradients'] or {}).items():
                            grouped.setdefault(group,[]).append(g)
                    summary['parameter_gradient_diagnostics']={group:dict(
                        measurements=len(gs),mean_weighted_ratio=statistics.mean(g['weighted_ratio'] for g in gs if g['weighted_ratio'] is not None),
                        mean_cosine=statistics.mean(g['cosine'] for g in gs if g['cosine'] is not None),
                        negative_cosine_fraction=statistics.mean(g['cosine']<0 for g in gs if g['cosine'] is not None))
                        for group,gs in grouped.items()}
                    atomic(path,summary)
            diag=sorted((directory/'loss_diagnostics').glob('window_*.json'))
            if diag:
                r=json.loads(diag[-1].read_text())
                if not r['optimizer_ran']:alerts.append(v+':latest_optimizer_step_skipped')
                latest[v]['latest_loss_diagnostics']=r
        for p in (directory/'previews').glob('*.json'):signature.append((v,p.name))
        for p in (directory/'milestones').glob('epoch_*/val.json'):signature.append((v,p.parent.name))
    free=shutil.disk_usage(root).free/1024**3
    if free<30:alerts.append('disk_below_30GiB')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv'],text=True,timeout=10)
    sig=json.loads(json.dumps([state['status'],signature]));sigpath=out/'plot_signature.json'
    oldsig=json.loads(sigpath.read_text()) if sigpath.exists() else None
    if _render_job is not None and _render_job.poll() is not None:
        if _render_job.returncode:
            _render_failed=True
            atomic(out/'render_error.json',dict(time=utc(),exit_code=_render_job.returncode))
        else:
            from scripts.analyze_conditional_operator import analyze
            analyze(root)
            atomic(sigpath,_render_signature)
            oldsig=_render_signature
            _render_failed=False
        _render_job=None
    if sig!=oldsig and latest and _render_job is None:
        with (out/'render.log').open('a') as log:
            _render_job=subprocess.Popen(['/home/ka/CKM/.venv/bin/python',str(REPO/'scripts/visualize_conditional_operator.py'),
                              '--root',str(root)],cwd=root/'code',stdout=log,stderr=subprocess.STDOUT)
        _render_signature=sig
    if _render_failed:alerts.append('visualization_failed')
    health=dict(updated_at=utc(),status=state['status'],alerts=alerts,jobs=state['jobs'],latest=latest,
                free_disk_gib=free,gpu=gpu,delivery='server files / journal; no chat push',
                rendering_pid=_render_job.pid if _render_job is not None else None)
    old=json.loads((out/'health.json').read_text()) if (out/'health.json').exists() else {}
    if alerts!=old.get('alerts') or state['status']!=old.get('status'):
        print(json.dumps(dict(time=utc(),status=state['status'],alerts=alerts)),flush=True)
    atomic(out/'health.json',health)
    report=f"# Conditional operator experiment monitor\n\nUpdated: {utc()}\n\nStatus: {state['status']}\n\n"
    report+=table(latest) if latest else 'Awaiting the first committed epoch.'
    report+='\n\nAlerts: '+('; '.join(alerts) if alerts else 'None')+'\n\n'
    report+='[Live plots and data](../visualizations/README.md)\n\nThe main loss remains FM MSE; see separate diagnostics for auxiliary terms and parameter gradients.'
    report+='\n\n[Interim Haar, gradient, and propagation-structure analysis](../visualizations/analysis.md)\n'
    report+='\nStatistics are saved every 100 epochs; Haar subbands are not electromagnetic propagation paths.\n'
    atomic(out/'latest.md',report)
    return terminal and 'visualization_failed' not in alerts and _render_job is None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();root=a.root.resolve();(root/'monitoring').mkdir(parents=True,exist_ok=True)
    with (root/'monitoring/monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                done=poll(root)
                if done or a.once:return
            except Exception as e:
                atomic(root/'monitoring/monitor_error.json',dict(time=utc(),error=repr(e)))
                print(repr(e),flush=True)
                if a.once:raise
            time.sleep(30)


if __name__=='__main__':main()
