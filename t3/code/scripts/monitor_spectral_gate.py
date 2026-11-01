"""Independent server heartbeat; no training mutations or silent restart."""
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

VARIANTS=('B','R','A','AP')


def numeric_means(rows):
    keys=set().union(*(r.keys() for r in rows)) if rows else set()
    return {k:statistics.mean(r[k] for r in rows if isinstance(r.get(k),(int,float)))
            for k in keys if any(isinstance(r.get(k),(int,float)) for r in rows)}


class Monitor:
    def __init__(self,root):
        self.root=root;self.out=root/'monitoring';self.out.mkdir(exist_ok=True)
        self.render=None;self.signature=None;self.failed_signature=None

    def poll(self):
        root,out=self.root,self.out
        if not (root/'suite_status.json').exists():
            atomic(out/'latest.md','# Spectral-gating experiment monitor\n\nPreparing the code snapshot and unit tests; no training results yet.\n');return False
        state=json.loads((root/'suite_status.json').read_text());terminal=state['status'] in ('complete','failed')
        alerts=[];latest={};signature=[]
        if not terminal and not process_matches(state.get('pid'),root):alerts.append('supervisor_missing')
        for v in VARIANTS:
            d=root/'8x8'/v;job=state['jobs'].get(v,{})
            if job.get('phase')=='failed':alerts.append(v+':job_failed')
            if not terminal and job.get('phase') in ('training','smoke','evaluate_val','evaluate_test'):
                if not process_matches(job.get('pid'),root):alerts.append(v+':process_missing')
                log=d/(job['phase']+'.log')
                # Training logs may be silent between milestones; check checkpoints too.
                mtimes=[p.stat().st_mtime for p in (log,d/'metrics.csv',d/'last.pt') if p.exists()]
                if mtimes and time.time()-max(mtimes)>1800:alerts.append(v+':stale_30min')
            rows,_=committed_history(d)
            if rows:
                end=rows[-1]['epoch'];latest[v]=summarize(rows,end)
                signature.append((v,min(end,1),end//20))
                windows=[]
                for p in (d/'loss_diagnostics').glob('window_*.json'):
                    r=json.loads(p.read_text())
                    if r['epoch_index']<end:windows.append(r)
                if windows:
                    last=max(windows,key=lambda r:r['optimizer_step_before'])
                    latest[v]['latest_output_gradients']=last['output_gradients']
                    latest[v]['cuda_peak_allocated_mib']=last.get('cuda_peak_allocated_mib')
                    if not last['optimizer_ran']:alerts.append(v+':optimizer_step_skipped')
                for epoch in range(100,end+1,100):
                    path=out/'statistics'/f'{v}_epoch_{epoch:04d}.json'
                    if path.exists():continue
                    block=[r for r in windows if epoch-100<=r['epoch_index']<epoch]
                    report=summarize(rows,epoch)
                    report.pop('fine_wavelet_and_feedback_gate',None)
                    report['losses_batch_mean']=numeric_means([r['losses'] for r in block])
                    report['skipped_windows']=sum(not r['optimizer_ran'] for r in block)
                    groups={};gates={}
                    for r in block:
                        for name,g in (r['parameter_gradients'] or {}).items():groups.setdefault(name,[]).append(g)
                        for name,g in (r.get('gate_diagnostics') or {}).items():
                            if g:gates.setdefault(name,[]).append(g)
                    report['parameter_gradients']={k:numeric_means(vs) for k,vs in groups.items()}
                    report['gate_diagnostics']={k:numeric_means(vs) for k,vs in gates.items()}
                    report['milestone']=f'../../8x8/{v}/milestones/epoch_{epoch:04d}/val.json'
                    atomic(path,report)
            for p in (d/'previews').glob('*.json'):signature.append((v,p.name))
            for p in (d/'milestones').glob('epoch_*/val.json'):signature.append((v,p.parent.name))
            for p in (d/'milestones').glob('epoch_*/gate_probe.json'):signature.append((v,p.parent.name,'gates'))
        sig=[state['status'],sorted(signature,key=str)]
        sig=json.loads(json.dumps(sig))
        sigpath=out/'plot_signature.json'
        old=json.loads(sigpath.read_text()) if sigpath.exists() else None
        if self.render is not None and self.render.poll() is not None:
            if self.render.returncode:
                self.failed_signature=self.signature
                atomic(out/'render_error.json',dict(time=utc(),exit_code=self.render.returncode))
            else:
                atomic(sigpath,self.signature);old=self.signature;self.failed_signature=None
            self.render=None
        if sig!=old and latest and self.render is None and sig!=self.failed_signature:
            with (out/'render.log').open('a') as log:
                self.render=subprocess.Popen([str(Path('/home/ka/CKM/.venv/bin/python')),str(root/'code/scripts/visualize_spectral_gate.py'),
                    '--root',str(root)],cwd=root/'code',stdout=log,stderr=subprocess.STDOUT,
                    env={**__import__('os').environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            self.signature=sig
        if self.failed_signature is not None:alerts.append('visualization_failed')
        free=shutil.disk_usage(root).free/1024**3
        if free<30:alerts.append('disk_below_30GiB')
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv'],text=True,timeout=10)
        health=dict(updated_at=utc(),status=state['status'],alerts=alerts,jobs=state['jobs'],latest=latest,
            free_disk_gib=free,gpu=gpu,delivery='server files and systemd journal; no external chat push',
            rendering_pid=self.render.pid if self.render else None)
        previous=json.loads((out/'health.json').read_text()) if (out/'health.json').exists() else {}
        if alerts!=previous.get('alerts') or state['status']!=previous.get('status'):
            print(json.dumps(dict(time=utc(),status=state['status'],alerts=alerts)),flush=True)
        atomic(out/'health.json',health)
        atomic(out/'latest.md',f"# B / R / A / AP experiment monitor\n\nUpdated: {utc()}\n\nStatus: {state['status']}\n\n"+
            (table(latest) if latest else 'Awaiting the first committed epoch.')+'\n\nAlerts: '+('; '.join(alerts) if alerts else 'None')+
            '\n\nHealth is checked every 30 seconds; statistics are recorded every 100 epochs. The table shows per-epoch N2 validation, not the primary N8 endpoint.\n\n'+
            '[Curves and fixed-scene visualizations](../visualizations/README.md) · [Frozen protocol](../protocol.md)\n\n'+
            'All four groups use the same MS source, CASA, and FM + 0.1 Haar. Radiomaps use jet; black denotes invalid/building pixels.\n')
        # On a rendering failure keep monitoring alive and report the error, no retry loop.
        return terminal and self.render is None and not self.failed_signature


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();m=Monitor(a.root.resolve())
    with (m.out/'monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if m.poll() or a.once:return
            except Exception as e:
                atomic(m.out/'monitor_error.json',dict(time=utc(),error=repr(e)))
                print(repr(e),flush=True)
                if a.once:raise
            time.sleep(30)


if __name__=='__main__':main()
