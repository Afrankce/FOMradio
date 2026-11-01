"""Durable AP no-drop experiment using the reference's immutable code snapshot."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.monitor_matern_suite import atomic, utc

OVERLAYS = ('run_no_drop.py', 'tests/test_no_drop.py',
            'scripts/run_no_drop_suite.py', 'scripts/monitor_no_drop.py',
            'scripts/visualize_no_drop.py', 'scripts/probe_no_drop.py')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare_config(root, reference):
    new = json.loads((root/'AP/_smoke/config.json').read_text())
    old = json.loads((reference/'8x8/AP/config.json').read_text())
    allowed = {'experiment','cfg_drop_prob','cfg_dropout_scope','source_sha256',
               'config_sha256','run_root','manifest_path','height_stats_path'}
    changes = {k: [old.get(k), new.get(k)] for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    unexpected = set(changes) - allowed
    if unexpected or new['cfg_drop_prob'] != 0. or old['cfg_drop_prob'] != .25:
        raise RuntimeError(f'Unexpected scientific configuration changes: {unexpected}')
    for k in ('initial_model_sha256','common_model_sha256','source_specification'):
        if new[k] != old[k]: raise RuntimeError(f'Unmatched {k}')
    atomic(root/'comparability.json', dict(passed=True, changes=changes,
        unchanged_initialization=True, same_source=True, same_seed=True,
        single_scientific_change='training condition-drop probability .25 -> 0'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(); root = a.root.resolve(); ref = a.reference.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        code = root/'code'
        if not code.exists():
            shutil.copytree(ref/'code', code, ignore=shutil.ignore_patterns('__pycache__'))
            for name in OVERLAYS:
                dst = code/name; dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO/name, dst)
            files = {str(f.relative_to(code)): sha(f) for f in code.rglob('*') if f.is_file()}
            atomic(root/'code_snapshot.json',dict(reference=str(ref/'code'),files=files,
                overlays=list(OVERLAYS),created_at=utc()))
        snapshot = json.loads((root/'code_snapshot.json').read_text())
        for name, digest in snapshot['files'].items():
            if sha(code/name) != digest: raise RuntimeError(f'Snapshot mutated: {name}')
        for name in ('manifest_samefreq_6.7ghz_8x8_0deg.jsonl','height_stats_train.json','scene_split_seed42.json'):
            dst = root/'manifests'/name; dst.parent.mkdir(exist_ok=True)
            if dst.exists():
                if sha(dst) != sha(ref/'manifests'/name): raise RuntimeError('Manifest changed')
            else: shutil.copy2(ref/'manifests'/name, dst)
        gpus = list(map(int,a.gpus.split(',')))
        if len(gpus)!=2 or len(set(gpus))!=2: raise ValueError('Exactly two training GPUs required')
        scope = dict(reference_root=str(ref),reference_run=str(ref/'8x8/AP'),
            training_gpus=gpus,variant='AP',cfg_drop_prob=0.,reference_cfg_drop_prob=.25,
            epochs=1000,seed=42,batch=128,world_size=2,source='MS',loss='FM + 0.1 Haar L2',
            cfg_scale=1.,ema_enabled=False,primary_steps=8,statistics_interval=100,
            final_test=True,feedback=False,internal_noise=False,
            source_pairing='unchanged scene/epoch counter-based x0 and t; independent loader generator',
            figures='descriptive single-seed, matched epoch; fixed jet [-300,0], errors [0,100]')
        if (root/'execution_scope.json').exists() and json.loads((root/'execution_scope.json').read_text())!=scope:
            raise RuntimeError('Execution scope changed')
        atomic(root/'execution_scope.json',scope)
        state = dict(status='preparing',pid=os.getpid(),started_at=utc(),jobs={})
        def update(phase, **kw):
            state['updated_at']=utc(); state['jobs']['AP']=dict(phase=phase,gpus=gpus,**kw)
            atomic(root/'suite_status.json',state); print(json.dumps(state),flush=True)
        update('unit_tests')
        if not (root/'unit_tests_passed.json').exists():
            with (root/'unit_tests.log').open('w') as log:
                subprocess.run([sys.executable,'-m','unittest','tests.test_no_drop','tests.test_spectral_gate',
                    'tests.test_no_ema','tests.test_detail_losses','tests.test_matern_source','-v'],
                    cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True)
            atomic(root/'unit_tests_passed.json',dict(passed=True,time=utc()))
        if a.prepare_only: return
        rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        for row in rows.splitlines():
            idx,mem,util=map(int,row.split(','))
            if idx in gpus and (mem>100 or util>0): raise RuntimeError(f'GPU {idx} is busy')
        atomic(root/'environment.json',dict(time=utc(),gpu=rows,python=sys.executable,
            packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()))
        out=root/'AP';out.mkdir(exist_ok=True)
        cfg=json.loads((ref/'8x8/AP/config.json').read_text())
        env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
        env.pop('PYTHONPATH',None)
        def execute(phase, args):
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                 str(code/'run_no_drop.py'),'--dataset-root',cfg['dataset_root'],
                 '--manifest-path',str(root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl'),
                 '--height-stats-path',str(root/'manifests/height_stats_train.json'),
                 '--run-root',str(out),'--variant','AP',*args]
            with (out/f'{phase}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
                update(phase,pid=child.pid,command=cmd)
                if child.wait(): raise RuntimeError(f'{phase} exit {child.returncode}')
        try:
            state['status']='smoke_testing'
            if not (out/'_smoke/smoke_verified.json').exists(): execute('smoke',['--smoke-optimizer-steps','2'])
            compare_config(root,ref)
            state['status']='running'
            runtime=out/'training_runtime.json'
            if not runtime.exists() or json.loads(runtime.read_text())['status']!='complete':
                execute('training',['--resume','auto'])
            for split in ('val','test'):
                execute('evaluate_'+split,['--evaluate',split])
            state['status']='complete';update('complete')
        except BaseException:
            state['status']='failed';update('failed',traceback=traceback.format_exc());raise


if __name__=='__main__': main()
