"""Four no-EMA runs and paired endpoint evaluation; frozen source and protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_matern_suite import snapshot, atomic, now, DATASET, MANIFESTS
from model.conditional_operator_uno import VARIANTS


class Suite:
    def __init__(self,root,code,gpus,reference):
        self.reference=reference
        self.root,self.code=root,code
        self.gpus={v:gpus[2*i:2*i+2] for i,v in enumerate(VARIANTS)}
        self.lock=threading.Lock()
        self.state=dict(status='smoke_testing',started_at=now(),pid=os.getpid(),jobs={})

    def update(self,v,phase,**extra):
        with self.lock:
            self.state['jobs'][v]=dict(phase=phase,gpus=self.gpus[v],updated_at=now(),**extra)
            self.state['updated_at']=now()
            atomic(self.root/'suite_status.json',self.state)

    def execute(self,v,phase,args):
        out=self.root/'8x8'/v;out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             str(self.code/'run_no_ema.py'),'--dataset-root',str(DATASET),
             '--manifest-path',str(self.root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl'),
             '--height-stats-path',str(self.root/'manifests/height_stats_train.json'),
             '--run-root',str(out),'--variant',v,*args]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,self.gpus[v])),
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
        env.pop('PYTHONPATH',None)
        with (out/f'{phase}.log').open('a') as log:
            p=subprocess.Popen(cmd,cwd=self.code,env=env,stdout=log,stderr=subprocess.STDOUT)
            self.update(v,phase,pid=p.pid,command=cmd)
            rc=p.wait()
        if rc:
            self.update(v,'failed',failed_phase=phase,exit_code=rc)
            raise RuntimeError(f'{v}/{phase} exit {rc}')

    def smoke(self,v):
        if not (self.root/'8x8'/v/'_smoke/smoke_verified.json').exists():
            self.execute(v,'smoke',['--smoke-optimizer-steps','2'])
        self.update(v,'smoke_passed')

    def reference_eval(self,v):
        for split in ('val','test'):
            self.execute(v,'reference_'+split,['--evaluate',split,'--reference-run',str(self.reference/'8x8'/v)])
        self.update(v,'reference_complete')

    def train(self,v):
        directory=self.root/'8x8'/v
        runtime=directory/'training_runtime.json'
        if not runtime.exists() or json.loads(runtime.read_text())['status']!='complete':
            self.execute(v,'training',['--resume','auto'])
        for split in ('val','test'):
            self.execute(v,'evaluate_'+split,['--evaluate',split])
        import torch
        from training.matern_source import digest_state
        before=torch.load(self.reference/'8x8'/v/'last.pt',map_location='cpu',weights_only=False)
        after=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
        same=digest_state(before['model'])==digest_state(after['model'])
        atomic(directory/'trajectory_audit.json',dict(endpoint_bitwise_equal=same,
            reference_online_sha256=digest_state(before['model']),online_sha256=digest_state(after['model']),
            interpretation='Compare paired old online/EMA for causal averaging effect; fresh runs supply online history.'))
        self.update(v,'complete')

    def parallel(self,fn):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fn,VARIANTS))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    gpus=list(map(int,a.gpus.split(',')))
    if len(gpus)!=8 or len(set(gpus))!=8:raise ValueError('Eight distinct idle GPU IDs required')
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code=snapshot(root)
        for name in ('manifest_samefreq_6.7ghz_8x8_0deg.jsonl','height_stats_train.json','scene_split_seed42.json'):
            dst=root/'manifests'/name;dst.parent.mkdir(exist_ok=True)
            if dst.exists():
                if dst.read_bytes()!=(MANIFESTS/name).read_bytes():raise RuntimeError('Manifest mismatch')
            else:shutil.copy2(MANIFESTS/name,dst)
        if not (root/'protocol.md').exists():shutil.copy2(code/'docs/NO_EMA_PILOT.md',root/'protocol.md')
        atomic(root/'execution_scope.json',dict(variants=list(VARIANTS),source='MS',gpus=gpus,
            epochs=1000,seed=42,auxiliary_weight=.1,baseline=str(a.reference.resolve()),reference=str(a.reference.resolve()),ema_enabled=False,
            initial_protocol_sha256=hashlib.sha256((root/'protocol.md').read_bytes()).hexdigest()))
        with (root/'unit_tests.log').open('w') as log:
            subprocess.run([sys.executable,'-m','unittest','tests.test_no_ema','tests.test_conditional_operator','tests.test_detail_losses','tests.test_matern_source','-v'],
                           cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True)
        if a.prepare_only:return
        gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        for row in gpu_rows.splitlines():
            i,mem,util=map(int,row.split(','))
            if i in gpus and (mem>1000 or util>5):raise RuntimeError(f'GPU {i} is no longer idle')
        atomic(root/'environment.json',dict(time=now(),python=sys.executable,gpus=gpu_rows,
            packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()))
        suite=Suite(root,code,gpus,a.reference.resolve())
        try:
            suite.state['status']='reference_evaluation'
            suite.parallel(suite.reference_eval)
            suite.state['status']='smoke_testing'
            suite.parallel(suite.smoke)
            import torch
            from training.matern_source import digest_state
            checks={}
            for v in VARIANTS:
                old=torch.load(a.reference/'8x8'/v/'_smoke/smoke.pt',map_location='cpu',weights_only=False)
                new=torch.load(root/'8x8'/v/'_smoke/smoke.pt',map_location='cpu',weights_only=False)
                checks[v]=dict(bitwise_equal=digest_state(old['model'])==digest_state(new['model']),
                    old_online=digest_state(old['model']),new_online=digest_state(new['model']))
            atomic(root/'smoke_trajectory_check.json',checks)
            if not all(c['bitwise_equal'] for c in checks.values()):
                raise RuntimeError('Smoke online weights changed: inspect before full training')
            configs=[json.loads((root/'8x8'/v/'_smoke/config.json').read_text()) for v in VARIANTS]
            if len({c['common_model_sha256'] for c in configs})!=1:
                raise RuntimeError('Common initial weights differ')
            if configs[2]['initial_model_sha256']!=configs[3]['initial_model_sha256']:
                raise RuntimeError('FILM and NSOP weights differ')
            atomic(root/'baseline_initialization_check.json',dict(passed=True,
                common_sha256=configs[0]['common_model_sha256'],
                parameter_budgets={v:configs[i] for i,v in enumerate(VARIANTS)}))
            suite.state['status']='running';suite.parallel(suite.train)
            suite.state['status']='complete';suite.state['updated_at']=now()
            atomic(root/'suite_status.json',suite.state)
        except Exception:
            suite.state['status']='failed';suite.state['traceback']=traceback.format_exc()
            atomic(root/'suite_status.json',suite.state);raise


if __name__=='__main__':main()
