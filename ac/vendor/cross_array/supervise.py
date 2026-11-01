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
from beam_boundary.common import atomic, read, sha, verify
from beam_boundary import supervise as shared


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root.resolve()
    verify(root);cfg=read(root/'config.json');assert cfg['experiment_kind']=='cross_array_observation'
    signal.signal(signal.SIGTERM,shared.stop);signal.signal(signal.SIGINT,shared.stop)
    with (root/'.supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not (root/'preflight.json').exists():
                atomic(root/'suite.json',dict(status='preflight',pid=os.getpid(),updated=time.time()))
                result=subprocess.run([sys.executable,'-m','unittest','test_cross_array','-v'],cwd=root/'code',capture_output=True,text=True)
                atomic(root/'unit_tests.json',dict(passed=result.returncode==0,output=result.stdout+result.stderr))
                if result.returncode: raise RuntimeError('Cross-array unit tests failed')
                for name,digest in read(root/'input_audit.json')['cache_sha256'].items():
                    if sha(root/'data'/name)!=digest: raise RuntimeError(f'Cache changed: {name}')
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for future in [pool.submit(shared.smoke,root,arm,'E0') for arm in ('T0','T2')]:future.result()
                atomic(root/'preflight.json',dict(passed=True,arms=['T0','T2'],resumed_updates='2+2 vs 4',updated=time.time()))
            atomic(root/'suite.json',dict(status='core_running',pid=os.getpid(),updated=time.time()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes=[f.result() for f in [pool.submit(shared.worker,root,arm,'E0') for arm in ('T0','T2')]]
            if not all(outcomes):raise RuntimeError('Training/validation failed; inspect arm logs')
            # Save the decision from validation BEFORE running either final test.
            subprocess.run([sys.executable,'-m','cross_array.analyze','--root',str(root),'--validation-only'],cwd=root/'code',check=True)
            atomic(root/'suite.json',dict(status='test_running',pid=os.getpid(),updated=time.time()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes=[f.result() for f in [pool.submit(shared.worker,root,arm,'E0',True) for arm in ('T0','T2')]]
            if not all(outcomes):raise RuntimeError('Test evaluation failed')
            subprocess.run([sys.executable,'-m','cross_array.analyze','--root',str(root)],cwd=root/'code',check=True)
            atomic(root/'suite.json',dict(status='complete',updated=time.time(),next_stage='Review progression_gate.json; no T3/fusion launched'))
        except Exception as error:
            atomic(root/'suite.json',dict(status='attention_required',error=str(error),traceback=traceback.format_exc(),updated=time.time()))
            print(traceback.format_exc(),flush=True)


if __name__=='__main__':main()
