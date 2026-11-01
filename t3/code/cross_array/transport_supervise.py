"""T3 only: one two-idle-GPU job, immutable completed T2 reference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from beam_boundary import supervise as shared
from beam_boundary.common import atomic, read, sha, verify
from .transport_prepare import verify_reference


def check_noise_pairing(root):
    parent=Path(read(root/'config.json')['parent_run'])
    for step in range(1,5):
        name=f'step_{step:06d}.json'
        old=read(parent/'jobs/T2/trace'/name);new=read(root/'preflight/T3-resumed/trace'/name)
        assert new['indices']==old['indices'] and new['t_sha256']==old['t_sha256']
        assert new['noise_sha256']==old['z_sha256']
        assert new['z_sha256']!=old['z_sha256']
    atomic(root/'preflight/T2_T3_pairing.json',dict(passed=True,steps=4,
        batch_and_time_identical=True,base_noise_identical=True,initial_state_changed=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root.resolve()
    verify(root);verify_reference(root);cfg=read(root/'config.json')
    assert cfg['experiment_kind']=='cross_array_transport' and cfg['new_training_arms']==['T3']
    signal.signal(signal.SIGTERM,shared.stop);signal.signal(signal.SIGINT,shared.stop)
    with (root/'.supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not (root/'preflight.json').exists():
                atomic(root/'suite.json',dict(status='preflight',pid=os.getpid(),updated=time.time()))
                unit=subprocess.run([sys.executable,'-m','unittest','test_cross_array','test_cross_array_transport','-v'],
                    cwd=root/'code',capture_output=True,text=True)
                atomic(root/'unit_tests.json',dict(passed=unit.returncode==0,output=unit.stdout+unit.stderr))
                if unit.returncode:raise RuntimeError('T3 unit tests failed')
                for name,digest in read(root/'input_audit.json')['cache_sha256'].items():
                    if sha(root/'data'/name)!=digest:raise RuntimeError(f'Input cache changed: {name}')
                shared.smoke(root,'T3','E0');check_noise_pairing(root)
                atomic(root/'preflight.json',dict(passed=True,arms=['T3'],resumed_updates='2+2 vs 4',paired_to='T2',updated=time.time()))
            atomic(root/'suite.json',dict(status='transport_running',pid=os.getpid(),updated=time.time()))
            if not shared.worker(root,'T3','E0'):raise RuntimeError('T3 training or validation failed')
            verify_reference(root)
            subprocess.run([sys.executable,'-m','cross_array.analyze','--root',str(root),'--validation-only'],cwd=root/'code',check=True)
            atomic(root/'suite.json',dict(status='test_running',pid=os.getpid(),updated=time.time()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                testing=pool.submit(shared.worker,root,'T3','E0',True)
                copying=pool.submit(subprocess.run,[sys.executable,'-m','cross_array.source_copy','--root',str(root)],
                    cwd=root/'code',check=True)
                copying.result()
                if not testing.result():raise RuntimeError('T3 final test failed')
            verify_reference(root)
            subprocess.run([sys.executable,'-m','cross_array.analyze','--root',str(root)],cwd=root/'code',check=True)
            atomic(root/'suite.json',dict(status='complete',updated=time.time(),next_stage='Review T3 versus T2; no new fusion/sigma/seed launched'))
        except Exception as error:
            atomic(root/'suite.json',dict(status='attention_required',error=str(error),traceback=traceback.format_exc(),updated=time.time()))
            print(traceback.format_exc(),flush=True)


if __name__=='__main__':main()
