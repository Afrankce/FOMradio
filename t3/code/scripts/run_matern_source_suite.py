"""Durable five-source pilot. No feedback or automatic seed/config expansion."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import sys
import threading
import traceback
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_matern_suite import snapshot,atomic,now,DATASET,MANIFESTS
from model.matern_source import SOURCES,SourceSampler,source_spectra

PLOT_PYTHON='/home/ka/CKM/.venv/bin/python'
MANIFESTS=Path('/home/ka/CKM/radio-unet/runs/width32_b128_20260908T052359Z/manifests')


def source_examples(root):
    if (root/'source_examples.npz').exists(): return
    meta=[dict(scene_id='source_visual_example',steering_deg=0.)]
    fields={};specs={}
    for name in SOURCES:
        sampler=SourceSampler(name)
        fields[name]=sampler.sample(meta,device='cpu',namespace='source_preview')[0,0].numpy()
        fields[name+'_covariance']=np.fft.irfft2(sampler.eigenvalues,s=(sampler.extent,sampler.extent))[0,:129]
        fields[name+'_spectrum']=sampler.eigenvalues
        specs[name]=sampler.specification()
    np.savez_compressed(root/'source_examples.npz',**fields)
    atomic(root/'sources.json',specs)


def gate(results):
    def values(name,key,n):
        rows=results[name]['records'];scenes=sorted({r['scene_id'] for r in rows})
        if len(scenes)!=80: raise RuntimeError('Gate needs 80 validation scenes')
        out=[]
        for scene in scenes:
            group=[r for r in rows if r['scene_id']==scene and r['steps']==n]
            if len(group)!=4 or {r['realization'] for r in group}!={0,1,2,3}:
                raise RuntimeError('Incomplete paired gate')
            out.append(np.mean([abs(r[key]) if key=='db_bias' else r[key] for r in group]))
        return scenes,np.array(out)
    selected=min(('S2','S8','S32'),key=lambda n:values(n,'fine_mse',8)[1].mean())
    comparisons=[]
    for baseline in ('W',selected):
        scenes,a=values(baseline,'fine_mse',8);ss,b=values('MS','fine_mse',8)
        if ss!=scenes: raise RuntimeError('Scene mismatch')
        difference=a-b;rng=np.random.default_rng(20260909)
        boot=np.array([difference[rng.integers(0,len(a),len(a))].mean() for _ in range(10000)])
        lo,hi=np.quantile(boot,[.025,.975]);fraction=float(difference.mean()/a.mean())
        guards=[]
        for n in (2,8):
            for key in ('db_rmse','db_bias'):
                sa,aa=values(baseline,key,n);sb,bb=values('MS',key,n)
                if sa!=sb: raise RuntimeError('Scene mismatch')
                guards.append(dict(steps=n,metric=key,delta=float((bb-aa).mean()),passed=bool((bb-aa).mean()<=.1)))
        comparisons.append(dict(baseline=baseline,candidate='MS',relative_fine_improvement=fraction,
            ci95=[float(lo),float(hi)],guardrails=guards,passed=bool(fraction>=.02 and lo>0 and all(g['passed'] for g in guards))))
    return dict(selected_single=selected,comparisons=comparisons,recommend_repeat_seeds=all(c['passed'] for c in comparisons),
                automatic_additional_training=False,training_seed=42,scope='validation only; selected-single CI descriptive')


class Supervisor:
    def __init__(self,root,code):
        self.root,self.code=root,code;self.lock=threading.Lock();self.stop=threading.Event()
        self.state=dict(status='preparing',started_at=now(),pid=os.getpid(),jobs={f'8x8/{v}':dict(phase='queued') for v in SOURCES})
        self.gpus=Queue()
        for i in range(4):self.gpus.put([2*i,2*i+1])

    def update(self,source,phase,**extra):
        with self.lock:
            self.state['jobs'][f'8x8/{source}']=dict(phase=phase,updated_at=now(),**extra)
            self.state['updated_at']=now();atomic(self.root/'suite_status.json',self.state)
            print(json.dumps(dict(source=source,phase=phase,**extra)),flush=True)

    def execute(self,source,gpus,phase,extra):
        directory=self.root/'8x8'/source;directory.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(self.code/'run_matern_source.py'),
             '--dataset-root',str(DATASET),'--manifest-path',str(self.root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl'),
             '--height-stats-path',str(self.root/'manifests/height_stats_train.json'),'--run-root',str(directory),'--source',source,*extra]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
             'PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'};env.pop('PYTHONPATH',None)
        with (directory/f'{phase}.log').open('a') as log:
            process=subprocess.Popen(cmd,cwd=self.code,env=env,stdout=log,stderr=subprocess.STDOUT)
            self.update(source,phase,pid=process.pid,gpus=gpus,command=cmd)
            status=process.wait()
        if status:
            self.stop.set();self.update(source,'failed',failed_phase=phase,exit_code=status)
            raise RuntimeError(f'{source} {phase} failed with {status}')

    def job(self,source,smoke):
        gpus=self.gpus.get()
        try:
            if self.stop.is_set(): self.update(source,'not_launched_after_failure');return
            directory=self.root/'8x8'/source
            if smoke:
                if not (directory/'_smoke/smoke_verified.json').exists():
                    self.execute(source,gpus,'smoke',['--smoke-optimizer-steps','2'])
                self.update(source,'smoke_passed',gpus=gpus)
            else:
                runtime=directory/'training_runtime.json'
                if not runtime.exists() or json.loads(runtime.read_text())['status']!='complete':
                    self.execute(source,gpus,'training',['--resume','auto'])
                for split in ('val','test'):
                    self.execute(source,gpus,f'evaluate_{split}',['--evaluate',split])
                self.update(source,'complete',gpus=gpus)
        finally:self.gpus.put(gpus)

    def parallel(self,smoke):
        errors=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks=[pool.submit(self.job,v,smoke) for v in ('W','S2','S8','MS','S32')]
            for future in as_completed(tasks):
                try:future.result()
                except Exception as error:errors.append(str(error));self.stop.set()
        if errors:raise RuntimeError('; '.join(errors))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code=snapshot(root)
        for name in ('manifest_samefreq_6.7ghz_8x8_0deg.jsonl','height_stats_train.json','scene_split_seed42.json'):
            target=root/'manifests'/name;target.parent.mkdir(exist_ok=True)
            if target.exists() and target.read_bytes()!=(MANIFESTS/name).read_bytes():raise RuntimeError('Manifest changed')
            if not target.exists():shutil.copy2(MANIFESTS/name,target)
        if not (root/'protocol.md').exists():shutil.copy2(code/'docs/MATERN_SOURCE_EXPERIMENT_PLAN.md',root/'protocol.md')
        if not (root/'execution_scope.json').exists():
            atomic(root/'execution_scope.json',dict(authorized_at=now(),array='8x8',sources=list(SOURCES),training_seeds=[42],
                feedback=False,confirmation='user explicitly deferred feedback to next round',epochs=1000,
                monitor_interval_seconds=30,statistics_interval_epochs=100,additional_seeds='recommendation_only',
                protocol_sha256=hashlib.sha256((root/'protocol.md').read_bytes()).hexdigest()))
        source_examples(root)
        if not (root/'unit_tests.log').exists():
            with (root/'unit_tests.log').open('w') as log:
                subprocess.run([sys.executable,'-m','unittest','tests.test_matern_source','-v'],cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True)
        if args.prepare_only:return
        atomic(root/'environment.json',dict(python=sys.executable,captured_at=now(),
            packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used','--format=csv'],text=True)))
        suite=Supervisor(root,code)
        try:
            suite.state['status']='smoke_testing';suite.parallel(True)
            configs=[json.loads((root/'8x8'/v/'_smoke/config.json').read_text()) for v in SOURCES]
            if len({c['initial_model_sha256'] for c in configs})!=1 or len({c['real_scalar_parameter_count'] for c in configs})!=1:
                raise RuntimeError('Source groups differ in initial architecture/weights')
            suite.state['status']='running_8x8';suite.parallel(False)
            results={v:json.loads((root/'8x8'/v/'val_diagnostics.json').read_text()) for v in SOURCES}
            atomic(root/'gate.json',gate(results))
            suite.state['status']='rendering_final'
            atomic(root/'suite_status.json',suite.state)
            with (root/'final_visualizations.log').open('a') as log:
                subprocess.run(['flock',str(root/'visualizations.lock'),PLOT_PYTHON,str(code/'scripts/visualize_source_suite.py'),'--root',str(root),'--final'],
                               cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True)
            suite.state['status']='complete';suite.state['updated_at']=now();atomic(root/'suite_status.json',suite.state)
        except Exception:
            suite.state['status']='failed_no_further_launch';suite.state['traceback']=traceback.format_exc()
            atomic(root/'suite_status.json',suite.state);raise


if __name__=='__main__':main()
