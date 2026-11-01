"""Four two-GPU slots, shared leases, conditional phases, bounded recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from .common import CORE, PHASE, atomic, read, verify, sha, checkpoint, phase_choice

STOP = False
CHILDREN = []


def stop(*_):
    global STOP
    STOP = True
    for child in list(CHILDREN):
        if child.poll() is None:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass


def idle_gpus():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
    occupied = set(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines())
    answer = []
    for row in rows:
        i, uuid, memory, use = [x.strip() for x in row.split(',')]
        if uuid not in occupied and int(memory) <= 100 and int(use) == 0: answer.append(int(i))
    return answer


def lease(root, arm):
    locks = root.parent / '.gpu_reservations'; locks.mkdir(exist_ok=True)
    allowed = read(root / 'config.json').get('allowed_gpu_ids')
    while not STOP:
        held = []
        for gpu in idle_gpus():
            if allowed is not None and gpu not in allowed: continue
            stream = (locks / f'gpu{gpu}.lock').open('a+')
            try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: stream.close(); continue
            if gpu not in idle_gpus(): stream.close(); continue
            held.append((gpu, stream))
            if len(held) == 2: return held
        for _, stream in held: stream.close()
        atomic(root / 'jobs' / arm / 'job.json', dict(arm=arm, phase='waiting_for_idle_gpus', updated=time.time()))
        time.sleep(10)
    raise RuntimeError('Stopping before GPU allocation')


def run(root, arm, gpus, module, arguments, label, retries=3, base='E2'):
    folder = root / 'jobs' / arm; log = folder / 'logs' / f'{label}.log'; log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m', module,
               '--root', str(root), '--arm', arm, '--base', base, *map(str, arguments)]
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), PYTHONPATH=str(root / 'code'),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', NCCL_ALGO='Ring', NCCL_PROTO='Simple')
    for attempt in range(1, retries + 1):
        if STOP: raise RuntimeError('Supervisor stopping')
        if shutil.disk_usage(root).free < 50 * 1024**3: raise RuntimeError('Disk guard: <50 GiB')
        verify(root)
        with log.open('a', buffering=1) as stream:
            stream.write(f'\nSTART {time.ctime()} attempt={attempt} {command}\n')
            child = subprocess.Popen(command, cwd=root / 'code', env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            CHILDREN.append(child)
            atomic(folder / 'job.json', dict(arm=arm, phase=label, pid=child.pid, gpus=gpus, attempt=attempt,
                                            log=str(log), base=base, updated=time.time()))
            began = time.time()
            try:
                while child.poll() is None:
                    if STOP: stop()
                    if shutil.disk_usage(root).free < 40 * 1024**3: raise RuntimeError('Emergency disk guard: <40 GiB')
                    updates = [began] + [p.stat().st_mtime for p in folder.glob('*progress*.json')]
                    # Smoke outputs are separate; smoke calls are bounded far below 30 min.
                    if time.time() - max(updates) > 1800: raise RuntimeError('No progress for 30 minutes')
                    time.sleep(5)
                if child.returncode == 0: return
                atomic(folder / 'last_failure.json', dict(phase=label, exit_code=child.returncode, attempt=attempt, updated=time.time()))
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=45)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                CHILDREN.remove(child)
        if attempt < retries: time.sleep(10)
    raise RuntimeError(f'{label} failed after {retries} attempts; see {log}')


def compare(left, right):
    import torch
    result = {}
    for name in ('model', 'exp_avg', 'exp_avg_sq'):
        aa = left['model'] if name == 'model' else {k: v[name] for k, v in left['optimizer']['state'].items()}
        bb = right['model'] if name == 'model' else {k: v[name] for k, v in right['optimizer']['state'].items()}
        maximum = 0.; numerator = 0.; denominator = 0.
        for key, v in bb.items():
            a = aa[key]
            if v.is_complex(): a, v = torch.view_as_real(a), torch.view_as_real(v)
            d = a.double() - v.double(); maximum = max(maximum, float(d.abs().max()))
            numerator += float(d.square().sum()); denominator += float(v.double().square().sum())
        relative = math.sqrt(numerator / max(1e-30, denominator)); result[name] = dict(max_absolute=maximum, relative_l2=relative)
        if name == 'model' and (maximum > 1e-5 or relative > 1e-5): raise RuntimeError(f'Model continuation failed: {result[name]}')
        if name != 'model' and relative > 1e-3: raise RuntimeError(f'Adam continuation failed: {result[name]}')
    if left['scaler'] != right['scaler'] or left['scheduler'] != right['scheduler']: raise RuntimeError('Scaler/scheduler continuation mismatch')
    return result


def smoke(root, arm, base='E2'):
    audit = root / 'preflight' / f'{arm}-continuation.json'
    if audit.exists() and read(audit)['passed']: return True
    held = lease(root, arm); gpus = [g for g, _ in held]
    try:
        resumed = root / 'preflight' / f'{arm}-resumed'; direct = root / 'preflight' / f'{arm}-direct'
        for step in (2, 4):
            run(root, arm, gpus, 'beam_boundary.train', ['--stop', step, '--out', resumed], f'smoke_resume_{step}', retries=1, base=base)
        run(root, arm, gpus, 'beam_boundary.train', ['--stop', 4, '--out', direct], 'smoke_direct_4', retries=1, base=base)
        left, lc = checkpoint(resumed); right, rc = checkpoint(direct); difference = compare(left, right)
        if not read(resumed / 'resume_verification_step_000002.json')['passed']: raise RuntimeError('Restore failed')
        for step in range(1, 5):
            if read(resumed / 'trace' / f'step_{step:06d}.json') != read(direct / 'trace' / f'step_{step:06d}.json'):
                raise RuntimeError('Batch/noise/time changed on restart')
        del left, right
        run(root, arm, gpus, 'beam_boundary.evaluate', ['--split', 'val', '--checkpoint-dir', resumed,
            '--tag', 'smoke', '--limit', 2, '--draws', 1, '--nfe', 2], 'smoke_eval', retries=1, base=base)
        atomic(audit, dict(passed=True, differences=difference, exact_streams=True,
            exact_state_restoration=True, bitwise_equal=lc['model_sha256'] == rc['model_sha256'], updated=time.time()))
        atomic(root / 'jobs' / arm / 'job.json', dict(arm=arm, phase='ready' if arm in CORE else 'conditional_wait', updated=time.time()))
        return True
    finally:
        for _, stream in held: stream.close()


def worker(root, arm, base='E2', test_only=False):
    cfg = read(root / 'config.json'); folder = root / 'jobs' / arm; held = lease(root, arm)
    try:
        gpus = [g for g, _ in held]
        if not test_only:
            for step in cfg['milestones']:
                committed = read(folder / 'committed.json')['step'] if (folder / 'committed.json').exists() else 0
                if committed < step: run(root, arm, gpus, 'beam_boundary.train', ['--stop', step], f'train_to_{step:06d}', base=base)
                committed = read(folder / 'committed.json')['step']
                if committed != step: continue
                destination = folder / 'evaluation' / f'step_{step:06d}' / 'val'
                if not (destination / 'complete.json').exists():
                    extra = [] if step == cfg['total_steps'] else ['--limit', cfg['preview_scenes'], '--draws', cfg['preview_draws']]
                    run(root, arm, gpus, 'beam_boundary.evaluate', ['--split', 'val', *extra], f'eval_{step:06d}_val', base=base)
            atomic(folder / 'job.json', dict(arm=arm, phase='validation_complete', step=cfg['total_steps'], gpus=gpus, base=base, updated=time.time()))
        else:
            dest = folder / 'evaluation' / f"step_{cfg['total_steps']:06d}" / 'test'
            if not (dest / 'complete.json').exists():
                run(root, arm, gpus, 'beam_boundary.evaluate', ['--split', 'test'], 'eval_final_test', base=base)
            atomic(folder / 'job.json', dict(arm=arm, phase='complete', step=cfg['total_steps'], gpus=gpus, base=base, updated=time.time()))
        return True
    except Exception as error:
        atomic(folder / 'job.json', dict(arm=arm, phase='failed', error=str(error), traceback=traceback.format_exc(), updated=time.time()))
        return False
    finally:
        for _, stream in held: stream.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); args = p.parse_args()
    root = args.root.resolve(); verify(root); signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with (root / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = read(root / 'config.json')
        if not (root / 'preflight.json').exists():
            atomic(root / 'suite.json', dict(status='preflight', pid=os.getpid(), updated=time.time()))
            unit = subprocess.run([sys.executable, '-m', 'unittest', 'test_beam_boundary', '-v'], cwd=root / 'code', capture_output=True, text=True)
            atomic(root / 'unit_tests.json', dict(passed=unit.returncode == 0, output=unit.stdout + unit.stderr))
            if unit.returncode: raise RuntimeError('Unit tests failed')
            if not read(root / 'architecture_audit.json')['passed']: raise RuntimeError('Architecture audit failed')
            # Verify all prepared input caches once before any formal updates.
            for filename, expected in read(root / 'input_audit.json')['cache_sha256'].items():
                if sha(root / 'data' / filename) != expected: raise RuntimeError(f'Cache changed: {filename}')
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(smoke, root, arm) for arm in CORE + PHASE]
                for future in futures: future.result()
            atomic(root / 'preflight.json', dict(passed=True, arms=list(CORE + PHASE), resumed_updates='2+2 vs 4', updated=time.time()))
        atomic(root / 'suite.json', dict(status='core_running', pid=os.getpid(), updated=time.time()))
        with ThreadPoolExecutor(max_workers=4) as pool:
            success = all([future.result() for future in [pool.submit(worker, root, arm) for arm in CORE]])
        if not success:
            atomic(root / 'suite.json', dict(status='attention_required', message='Core arm failed; phase selection blocked', updated=time.time())); return
        selection_path = root / 'phase_selection.json'
        if selection_path.exists(): selection = read(selection_path)
        else:
            paths = {a: root / 'jobs' / a / 'evaluation' / f"step_{cfg['total_steps']:06d}" / 'val/summary.json' for a in CORE}
            scores = {a: read(path) for a, path in paths.items()}
            selection = dict(base=phase_choice(scores), decision='run' if phase_choice(scores) else 'conditional_skip',
                validation_only=True, scores=scores, validation_sha256={a: sha(path) for a, path in paths.items()}, updated=time.time())
            atomic(selection_path, selection)
        if selection['base'] is None:
            for arm in PHASE: atomic(root / 'jobs' / arm / 'job.json', dict(arm=arm, phase='conditional_skipped',
                reason='No spectral candidate passed validation delta improvement and map guardrail', updated=time.time()))
        atomic(root / 'suite.json', dict(status='phase_and_test_running', pid=os.getpid(), phase_base=selection['base'], updated=time.time()))
        with ThreadPoolExecutor(max_workers=4) as pool:
            # Both phases queued first; spare slots finalize core tests.
            phase_futures = [pool.submit(worker, root, arm, selection['base']) for arm in PHASE] if selection['base'] else []
            tests = [pool.submit(worker, root, arm, 'E2', True) for arm in CORE]
            phase_ok = all([f.result() for f in phase_futures])
            success = all([f.result() for f in tests]) and phase_ok
            if selection['base'] and phase_ok:
                success = all([f.result() for f in [pool.submit(worker, root, arm, selection['base'], True) for arm in PHASE]]) and success
        if success:
            result = subprocess.run([sys.executable, '-m', 'beam_boundary.analyze', '--root', str(root)], cwd=root / 'code')
            success = result.returncode == 0
        atomic(root / 'suite.json', dict(status='complete' if success else 'attention_required', phase_base=selection['base'], updated=time.time()))


if __name__ == '__main__': main()
