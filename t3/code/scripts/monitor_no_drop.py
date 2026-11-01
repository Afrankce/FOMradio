"""Server-owned heartbeat, paired 100-epoch reports and isolated diagnostics."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.monitor_matern_suite import atomic,utc,process_matches,committed_history,summarize,table

PY='/home/ka/anaconda3/envs/radio-unet-w32/bin/python'
PLOT='/home/ka/CKM/.venv/bin/python'


class Monitor:
    def __init__(self,root):
        self.root=root;self.out=root/'monitoring';self.out.mkdir(exist_ok=True)
        self.render=None;self.probe=None;self.signature=None;self.failed_render=None;self.failed_probes=set()
        self.scope=json.loads((root/'execution_scope.json').read_text())
        self.ref=Path(self.scope['reference_run'])

    def poll(self):
        root,out=self.root,self.out
        state=json.loads((root/'suite_status.json').read_text());terminal=state['status'] in ('complete','failed')
        alerts=[]
        if state['status']=='failed':alerts.append('training_or_evaluation_failed')
        if not terminal and not process_matches(state.get('pid'),root):alerts.append('supervisor_missing')
        job=state.get('jobs',{}).get('AP',{})
        if job.get('pid') and not terminal and not process_matches(job['pid'],root):alerts.append('worker_missing')
        d=root/'AP';rows,_=committed_history(d);old,_=committed_history(self.ref)
        end=rows[-1]['epoch'] if rows else 0
        paths=[p for p in (d/'metrics.csv',d/'last.pt',d/(job.get('phase','training')+'.log')) if p.exists()]
        if paths and not terminal and time.time()-max(p.stat().st_mtime for p in paths)>1800:alerts.append('no_progress_30min')
        latest={}
        if rows:latest={'NoDrop':summarize(rows,end),'Drop025 (matched epoch)':summarize(old,end)}
        windows=[]
        for p in (d/'loss_diagnostics').glob('window_*.json'):
            r=json.loads(p.read_text())
            if r['epoch_index']<end:windows.append(r)
        if windows:
            last=max(windows,key=lambda r:r['optimizer_step_before'])
            if not last['optimizer_ran']:alerts.append('latest_optimizer_step_skipped')
        for epoch in range(100,end+1,100):
            file=out/'statistics'/f'epoch_{epoch:04d}.json'
            milestone=d/'milestones'/f'epoch_{epoch:04d}'/'val.json'
            if file.exists() or not milestone.exists():continue
            blocks=[r for r in windows if epoch-100<=r['epoch_index']<epoch]
            gradients=[r['parameter_gradients'] for r in blocks if r['parameter_gradients']]
            stats=dict(epoch=epoch,nodrop=summarize(rows,epoch),reference=summarize(old,epoch),
                skipped_windows=sum(not r['optimizer_ran'] for r in blocks),
                mean_train_loss_terms={k:statistics.mean(r['losses'][k] for r in blocks) for k in blocks[0]['losses']} if blocks else {},
                gradient_observations=len(gradients),
                nodrop_validation=json.loads(milestone.read_text())['aggregated'],
                reference_validation=json.loads((self.ref/'milestones'/f'epoch_{epoch:04d}'/'val.json').read_text())['aggregated'])
            stats['val_N2_rmse_delta_db']=stats['nodrop']['last']['val_db_rmse']-stats['reference']['last']['val_db_rmse']
            atomic(file,stats);print(f'{utc()} completed paired epoch {epoch} report',flush=True)
        if self.probe is not None and self.probe.poll() is not None:
            if self.probe.returncode:
                self.failed_probes.add(self.probe_epoch)
                atomic(out/'probe_error.json',dict(epoch=self.probe_epoch,exit_code=self.probe.returncode,time=utc()))
            self.probe=None
        pending=[e for e in range(100,end+1,100) if (d/'milestones'/f'epoch_{e:04d}'/'val.json').exists()
                 and not (out/'probes'/f'epoch_{e:04d}.json').exists() and e not in self.failed_probes]
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
        if pending and self.probe is None:
            available=[i for i,m,u in (map(int,l.split(',')) for l in gpu.splitlines())
                       if i not in self.scope['training_gpus'] and m<100 and u==0]
            if available:
                self.probe_epoch=pending[0]
                with (out/'probe.log').open('a') as log:
                    self.probe=subprocess.Popen([PY,str(root/'code/scripts/probe_no_drop.py'),'--root',str(root),'--epoch',str(pending[0])],
                        cwd=root/'code',stdout=log,stderr=subprocess.STDOUT,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(available[0]),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        signature=[end//20,min(end,1),state['status'],
            sorted(str(p.relative_to(root)) for pattern in ('AP/previews/*.json','AP/milestones/epoch_*/val.json',
                'AP/test_diagnostics.json','monitoring/probes/*.json') for p in root.glob(pattern))]
        sigpath=out/'plot_signature.json';previous=json.loads(sigpath.read_text()) if sigpath.exists() else None
        if self.render is not None and self.render.poll() is not None:
            if self.render.returncode:
                self.failed_render=self.signature
                atomic(out/'render_error.json',dict(exit_code=self.render.returncode,signature=self.signature,time=utc()))
            else:atomic(sigpath,self.signature);previous=self.signature;self.failed_render=None
            self.render=None
        if end and self.render is None and signature!=previous and signature!=self.failed_render:
            with (out/'render.log').open('a') as log:
                self.render=subprocess.Popen([PLOT,str(root/'code/scripts/visualize_no_drop.py'),'--root',str(root)],
                    cwd=root/'code',stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            self.signature=signature
        if self.failed_render is not None:alerts.append('visualization_failed')
        if self.failed_probes:alerts.append('gradient_probe_failed')
        free=shutil.disk_usage(root).free/1024**3
        if free<30:alerts.append('disk_below_30GiB')
        health=dict(status=state['status'],updated_at=utc(),epoch=end,alerts=alerts,jobs=state.get('jobs'),
            latest=latest,gpu=gpu,free_disk_gib=free,rendering_pid=self.render.pid if self.render else None,
            probe_pid=self.probe.pid if self.probe else None,pending_probes=pending,
            delivery='server files/systemd journal; no app push automation available')
        prior=json.loads((out/'health.json').read_text()) if (out/'health.json').exists() else {}
        if alerts!=prior.get('alerts') or state['status']!=prior.get('status'):
            print(json.dumps(dict(time=utc(),status=state['status'],alerts=alerts)),flush=True)
        atomic(out/'health.json',health)
        atomic(out/'latest.md','# AP No-drop comparison monitor\n\n'+f'Updated: {utc()}\n\nStatus: {state["status"]}; alerts: {alerts or "None"}\n\n'+
            (table(latest) if latest else 'Awaiting the first committed epoch.')+
            '\n\nThe table shows N2 full-condition validation. Training losses correspond to different tasks and should not be ranked directly. Health is checked every 30 seconds; paired statistics and N8 evaluation run every 100 epochs.\n\n'+
            '[Visualizations](../visualizations/README.md) · [Protocol](../protocol.md) · [Comparability](../comparability.json)\n\n'+
            'Training is managed by server systemd and does not depend on the local client. Alerts are saved to files and the journal; there is no external chat push.\n')
        return terminal and self.render is None and self.probe is None and not pending and not alerts


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();m=Monitor(a.root.resolve())
    with (m.out/'monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if m.poll() or a.once:return
            except Exception as e:
                atomic(m.out/'monitor_error.json',dict(time=utc(),error=repr(e)));print(repr(e),flush=True)
                if a.once:raise
            time.sleep(30)


if __name__=='__main__':main()
