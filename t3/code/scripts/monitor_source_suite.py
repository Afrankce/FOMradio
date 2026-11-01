"""Server-side source-suite heartbeat and real 100-epoch diagnostic summaries."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import statistics

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.monitor_matern_suite import atomic,utc,process_matches,committed_history,summarize,table
SOURCES=('W','S2','S8','S32','MS')


def poll(root):
    out=root/'monitoring';out.mkdir(exist_ok=True)
    path=root/'suite_status.json'
    if not path.exists():
        atomic(out/'latest.md','# Source-distribution experiment monitor\n\nAwaiting server job initialization.\n');return False
    state=json.loads(path.read_text());terminal=state['status'] in ('complete','failed_no_further_launch')
    alerts=[];latest={};jobs={};diagnostics={};signature=[]
    if not terminal and not process_matches(state.get('pid'),root):alerts.append('supervisor_missing')
    for name in SOURCES:
        key=f'8x8/{name}';job=state['jobs'].get(key,dict(phase='queued'));directory=root/key
        info=dict(job)
        if job['phase'] in ('training','smoke','evaluate_val','evaluate_test'):
            if not process_matches(job.get('pid'),root) and not terminal:alerts.append(key+':process_missing')
            candidates=[directory/(job['phase']+'.log'),directory/'committed_epoch.json']
            timestamps=[p.stat().st_mtime for p in candidates if p.exists()]
            age=time.time()-max(timestamps) if timestamps else 0
            info['progress_age_seconds']=round(age,1)
            if age>1800 and not terminal:alerts.append(key+':stalled_30min')
        if job['phase']=='failed':alerts.append(key+':failed')
        try:
            rows,_=committed_history(directory)
            if rows:
                latest[key]=summarize(rows,rows[-1]['epoch']);info['epoch']=rows[-1]['epoch']
                signature.append((name,rows[-1]['epoch']//100))
                for epoch in range(100,rows[-1]['epoch']+1,100):
                    target=out/key/f'epoch_{epoch:04d}.json'
                    metrics=directory/'milestones'/f'epoch_{epoch:04d}'/'val.json'
                    if not metrics.exists():continue
                    signature.append((name,epoch,'diagnostic'))
                    data=json.loads(metrics.read_text());summary={}
                    for n in (2,8):
                        selected=[r for r in data['records'] if r['steps']==n]
                        summary[str(n)]={k:statistics.mean(r[k] for r in selected) for k in ('db_rmse','fine_mse','raw_db_rmse')}
                    diagnostics[key]=dict(epoch=epoch,metrics=summary)
                    if target.exists():continue
                    base=summarize(rows,epoch);base.pop('fine_wavelet_and_feedback_gate',None)
                    atomic(target,dict(**base,actual_validation=summary,identity=data['identity']))
                    report=f'# {key} — epoch {epoch}\n\n'+table({key:base})+'\n\n'
                    report+='| Euler N | Scene-mean RMSE (dB) | Fine Haar MSE | Raw RMSE (dB) |\n|---|---:|---:|---:|\n'
                    for n,values in summary.items():report+=f"| {n} | {values['db_rmse']:.5f} | {values['fine_mse']:.5f} | {values['raw_db_rmse']:.5f} |\n"
                    report+='\n80 validation scenes x 4 realizations; one training seed, not test metrics.\n'
                    atomic(target.with_suffix('.md'),report)
                    print(json.dumps(dict(event='100epoch_statistics',job=key,epoch=epoch,report=str(target.with_suffix('.md')))),flush=True)
        except (OSError,ValueError,RuntimeError,KeyError) as error:
            info['read_warning']=str(error)
        jobs[key]=info
    free=shutil.disk_usage(root).free/1024**3
    if free<30:alerts.append('disk_free_below_30GiB')
    try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv'],text=True,timeout=10)
    except Exception as error:gpu=str(error);alerts.append('gpu_telemetry_error')
    old=json.loads((out/'health.json').read_text()) if (out/'health.json').exists() else {}
    health=dict(updated_at=utc(),status=state['status'],jobs=jobs,alerts=alerts,free_disk_gib=free,gpu=gpu,
                notification_delivery='server files and systemd journal; no chat push',poll_seconds=30,statistics_every_epochs=100)
    atomic(out/'health.json',health)
    if old.get('alerts')!=alerts:print(json.dumps(dict(event='health_change',alerts=alerts)),flush=True)
    report=f"# External Matérn source-distribution monitor\n\nUpdated: {health['updated_at']}\n\nStatus: {state['status']}\n\n"
    report+=(table(latest) if latest else 'No committed epoch yet.')
    report+='\n\nQueued/current jobs:\n\n'
    for name,job in jobs.items():report+=f"- {name}: {job['phase']}; GPU {job.get('gpus',[])}.\n"
    report+='\nCurrent metrics from different epochs are not a fair ranking. Detail reports every 100 epochs use actual validation inference; comparison plots are generated once all five groups reach the same checkpoint.\n'
    report+='\nAlerts: '+('; '.join(alerts) if alerts else 'None')+'\n'
    report+=f'\n[Visualization index]({root}/visualizations/README.md)\n'
    atomic(out/'latest.md',report)
    signature_file=out/'plot_signature.json';sig=json.loads(json.dumps(signature))
    previous=json.loads(signature_file.read_text()) if signature_file.exists() else None
    if sig!=previous and not terminal and state['status']!='rendering_final':
        cmd=['flock',str(root/'visualizations.lock'),'/home/ka/CKM/.venv/bin/python',str(root/'code/scripts/visualize_source_suite.py'),'--root',str(root)]
        with (out/'visualizations.log').open('a') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        if result.returncode:
            atomic(out/'visualization_error.json',dict(time=utc(),exit_code=result.returncode))
            print(json.dumps(dict(event='visualization_failed',exit_code=result.returncode)),flush=True)
        else:atomic(signature_file,sig)
    return terminal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();root=args.root.resolve();(root/'monitoring').mkdir(exist_ok=True)
    with (root/'monitoring/monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if poll(root) or args.once:return
            except Exception as error:
                atomic(root/'monitoring/monitor_error.json',dict(time=utc(),error=repr(error)))
                print(repr(error),flush=True)
                if args.once:raise
            time.sleep(30)


if __name__=='__main__':main()
