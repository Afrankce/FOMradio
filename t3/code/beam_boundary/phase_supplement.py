"""Prepare and supervise only the explicitly requested supplementary phase pair."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from .common import PHASE, atomic, read, sha, verify, state_sha
from . import supervise


def audit(base):
    import torch
    from .model import BoundaryUNO
    torch.set_num_threads(2); torch.manual_seed(42)
    reference = BoundaryUNO(base).state_dict(); result = {}; states = []
    for arm in PHASE:
        torch.manual_seed(42); model = BoundaryUNO(arm, base, phase_input_only=True)
        state = model.state_dict()
        assert all(torch.equal(value, state[key]) for key, value in reference.items())
        states.append(state_sha(state))
        result[arm] = dict(real_parameters=sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()),
                           initial_sha256=states[-1], common_base_sha256=state_sha(reference))
    assert len(set(states)) == 1
    return dict(passed=True, base=base, identical_initial_states=True, arms=result)


def prepare(root, parent, project):
    if root.exists(): raise RuntimeError('Refusing to overwrite an experiment directory')
    verify(parent)
    previous_selection = read(parent / 'phase_selection.json')
    scores = {a: read(parent / 'jobs' / a / 'evaluation/step_005000/val/summary.json') for a in ('E2', 'E3', 'E5')}
    base = min(scores, key=lambda a: (scores[a]['unseen_delta_rmse_db'], scores[a]['map_rmse_db'], a))
    root.mkdir(parents=True)
    shutil.copytree(parent / 'code', root / 'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(project / 'beam_boundary', root / 'code/beam_boundary', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('test_beam_boundary.py', 'test_beam_phase_supplement.py'):
        shutil.copy2(project / 'tests' / name, root / 'code' / name)
    shutil.copy2(project / 'docs/BEAM_PHASE_SUPPLEMENT.md', root / 'PROTOCOL.md')
    # A read-only mmap consumer; never alter the shared verified numeric cache.
    (root / 'data').symlink_to(parent / 'data', target_is_directory=True)
    shutil.copy2(parent / 'input_audit.json', root / 'input_audit.json')
    cfg = read(parent / 'config.json')
    cfg.update(schema=2, arms=list(PHASE), phase_arms=[], display_arms=['E0', base, *PHASE],
        experiment_kind='phase_supplement', parent_run=str(parent), phase_base=base, phase_input_only=True,
        phase_selection='explicit_user_override_min_validation_among_E2_E3_E5_no_threshold',
        comparison_pairs=[['P-A', base], ['P-AP', 'P-A'], ['P-AP', base], ['P-A', 'E0'], ['P-AP', 'E0']],
        created=time.time(), new_training_arms=list(PHASE), reference_arms=['E0', base])
    atomic(root / 'config.json', cfg)
    atomic(root / 'phase_selection.json', dict(base=base, decision='explicit_user_requested_supplement',
        previous_decision=previous_selection['decision'], previous_run=str(parent),
        previous_selection_sha256=sha(parent / 'phase_selection.json'),
        rationale='Minimum prior validation error among spectral candidates; no test-based choice',
        exploratory_after_prior_results=True, previous_run_unchanged=True,
        scores={a: {k: v[k] for k in ('map_rmse_db', 'unseen_delta_rmse_db')} for a, v in scores.items()},
        validation_sha256={a: sha(parent / 'jobs' / a / 'evaluation/step_005000/val/summary.json') for a in scores}))
    references = {}
    for arm in ('E0', base):
        src = parent / 'jobs' / arm; dst = root / 'jobs' / arm; dst.mkdir(parents=True)
        for name in ('history.json', 'progress.json', 'committed.json'):
            shutil.copy2(src / name, dst / name)
        for summary in src.glob('evaluation/step_*/*/summary.json'):
            target = dst / summary.relative_to(src); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary, target); shutil.copy2(summary.parent / 'complete.json', target.parent / 'complete.json')
            references[str(target.relative_to(root))] = dict(source=str(summary), sha256=sha(summary))
        atomic(dst / 'job.json', dict(arm=arm, phase='reference_only_completed', step=5000,
                                     source_run=str(parent), updated=time.time()))
    atomic(root / 'references.json', references)
    for arm in PHASE: atomic(root / 'jobs' / arm / 'job.json', dict(arm=arm, phase='queued', updated=time.time()))
    atomic(root / 'snapshot.json', dict(code_sha256={str(p.relative_to(root / 'code')): sha(p)
        for p in sorted((root / 'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
        parent_snapshot_sha256=sha(parent / 'snapshot.json'), protocol_sha256=sha(root / 'PROTOCOL.md')))
    result = subprocess.run([sys.executable, '-c',
        f'from beam_boundary.phase_supplement import audit; import json; print(json.dumps(audit({base!r})))'],
        cwd=root / 'code', capture_output=True, text=True, check=True)
    atomic(root / 'architecture_audit.json', __import__('json').loads(result.stdout.splitlines()[-1]))
    atomic(root / 'suite.json', dict(status='prepared', phase_base=base, updated=time.time()))
    print(root, flush=True)


def launch(root):
    verify(root); cfg = read(root / 'config.json'); base = cfg['phase_base']
    assert cfg['experiment_kind'] == 'phase_supplement' and cfg['phase_input_only']
    signal.signal(signal.SIGTERM, supervise.stop); signal.signal(signal.SIGINT, supervise.stop)
    with (root / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if not (root / 'preflight.json').exists():
                atomic(root / 'suite.json', dict(status='preflight', pid=os.getpid(), phase_base=base, updated=time.time()))
                result = subprocess.run([sys.executable, '-m', 'unittest', 'test_beam_boundary', 'test_beam_phase_supplement', '-v'],
                    cwd=root / 'code', capture_output=True, text=True)
                atomic(root / 'unit_tests.json', dict(passed=result.returncode == 0, output=result.stdout + result.stderr))
                if result.returncode: raise RuntimeError('Phase unit tests failed')
                if not read(root / 'architecture_audit.json')['passed']: raise RuntimeError('Architecture audit failed')
                for name, expected in read(root / 'input_audit.json')['cache_sha256'].items():
                    if sha(root / 'data' / name) != expected: raise RuntimeError(f'Cache changed: {name}')
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for future in [pool.submit(supervise.smoke, root, arm, base) for arm in PHASE]: future.result()
                atomic(root / 'preflight.json', dict(passed=True, arms=list(PHASE), base=base,
                    resumed_updates='2+2 vs 4', phase_input_only=True, updated=time.time()))
            atomic(root / 'suite.json', dict(status='phase_running', pid=os.getpid(), phase_base=base, updated=time.time()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = [f.result() for f in [pool.submit(supervise.worker, root, arm, base) for arm in PHASE]]
            if not all(outcomes): raise RuntimeError('A phase validation worker failed; inspect job logs')
            atomic(root / 'suite.json', dict(status='test_running', pid=os.getpid(), phase_base=base, updated=time.time()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = [f.result() for f in [pool.submit(supervise.worker, root, arm, base, True) for arm in PHASE]]
            if not all(outcomes): raise RuntimeError('A phase test worker failed; inspect job logs')
            subprocess.run([sys.executable, '-m', 'beam_boundary.analyze', '--root', str(root)], cwd=root / 'code', check=True)
            atomic(root / 'suite.json', dict(status='complete', phase_base=base, updated=time.time()))
        except Exception as error:
            atomic(root / 'suite.json', dict(status='attention_required', error=str(error),
                traceback=traceback.format_exc(), phase_base=base, updated=time.time()))
            # Scientific failures remain visible and do not restart indefinitely.
            print(traceback.format_exc(), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--parent', type=Path)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.action == 'prepare':
        if not args.parent: parser.error('--parent required for prepare')
        prepare(args.root.resolve(), args.parent.resolve(), args.project.resolve())
    else: launch(args.root.resolve())


if __name__ == '__main__': main()
