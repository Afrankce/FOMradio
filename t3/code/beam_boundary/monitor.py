"""Independent CPU monitor and durable dashboard; no client push assumed."""
import argparse
import html
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from .common import CORE, PHASE, LABELS, atomic, atomic_text, read


def optional(path, default=None): return read(path) if path.exists() else default


def tick(root):
    now = time.time(); suite = optional(root / 'suite.json', {}); jobs = {}; alerts = []
    cfg = read(root / 'config.json'); arms = cfg.get('display_arms', CORE + PHASE)
    supplement = cfg.get('experiment_kind') == 'phase_supplement'
    transport = cfg.get('experiment_kind') == 'cross_array_transport'
    cross_array = cfg.get('experiment_kind') in ('cross_array_observation','cross_array_transport')
    if suite.get('status') == 'attention_required': alerts.append('Supervisor: ' + suite.get('error', suite.get('message', 'attention required')))
    for arm in arms:
        folder = root / 'jobs' / arm; job = optional(folder / 'job.json', {})
        progress = optional(folder / 'progress.json', {}); commit = optional(folder / 'committed.json', {})
        job.update(progress=progress, committed_step=commit.get('step', 0)); jobs[arm] = job
        if job.get('phase') == 'failed': alerts.append(f"{arm}: {job.get('error')}")
        if job.get('pid') and now - job.get('updated', now) > 90:
            try: os.kill(job['pid'], 0)
            except ProcessLookupError: alerts.append(f'{arm}: recorded child PID missing')
        if job.get('phase', '').startswith(('train_to', 'eval_')):
            timestamps = [job.get('updated', now)] + [p.stat().st_mtime for p in folder.glob('*progress*.json')]
            if now - max(timestamps) > 600: alerts.append(f'{arm}: no progress for over 10 minutes')
    if suite.get('pid') and suite.get('status') not in ('complete', 'attention_required'):
        try: os.kill(suite['pid'], 0)
        except ProcessLookupError: alerts.append('Supervisor is absent; check persistent service')
    free = shutil.disk_usage(root).free / 1024**3
    if free < 60: alerts.append(f'Free disk only {free:.1f} GiB')
    renderer = optional(root / 'render_job.json', {})
    if renderer.get('exit_code') not in (None, 0): alerts.append('Visualization failed; training is independent; see render.log')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu', '--format=csv,noheader,nounits'], text=True)
    state = dict(updated=now, time=time.strftime('%Y-%m-%d %H:%M:%S %Z'), suite=suite, jobs=jobs, alerts=alerts, gpus=gpu, disk_free_gib=free)
    atomic(root / 'monitor.json', state)
    if alerts != optional(root / 'alerts_latest.json', {}).get('alerts'):
        atomic(root / 'alerts' / f'{int(now)}.json', dict(updated=now, alerts=alerts))
    atomic(root / 'alerts_latest.json', dict(updated=now, alerts=alerts))
    heading = 'Independent P-A / P-AP supplement' if supplement else 'Beam-boundary conditional-operator monitor'
    selection_note = (f"The user requested this additional run. Both arms use base {cfg['phase_base']} and the same complex output head; only phase input differs. Training restarts from seed 42. E0/{cfg['phase_base']} is a reference from the prior round, not newly trained here."
        if supplement else 'P-A/P-AP are conditional experiments. Complete six core validations first, then select the base by the preregistered gate. The test set does not influence selection.')
    gate_note = ('The old launch gate no longer applies in this round; prior skip records remain in the previous round. This supplement is exploratory because earlier results were observed.'
        if supplement else 'Phase gate: among E2/E3/E5, validation unseen deltaRMSE must improve at least 1% over E0, while map RMSE is no worse than 1.01 times E0; otherwise skip and record.')
    training_note = 'Seed 42; two GPUs per arm, at most four arms concurrently; 5000 successful updates; original NoDrop AP/MS-Matérn/FM+Haar with added 64x64 residual.'
    if cross_array:
        heading = 'T0 / T2: 8x8 observed condition to 32x32 array target'
        selection_note = 'Source Y8 is a complete observed map from the same scene at fixed 0 degrees; target angle changes. T0 hides source values and validity, while T2 reads them. Parameters and initialization match. This is not spatial super-resolution.'
        training_note = 'Seed 42; two free GPUs per arm, restricted to GPUs 4-7; 5000 successful updates; original NoDrop AP U-NO/MS-Matérn/FM+Haar; no new boundary or phase branch.'
        gate_note = 'Final validation only: unseen-angle response improves at least 1%, native near-building RMSE and shadow-transition gradient each improve at least 3%, paired-scene interval upper bound is below zero, and overall error does not worsen by more than 1%. T3/new fusion is not launched yet.'
    if transport:
        heading = 'T3: cross-array FM initialized from observed Y8 plus noise'
        selection_note = 'T2 is a completed reference from the prior round and is not retrained. Only T3 is newly trained: x0=mu8+z and target velocity Y32-mu8-z; conditions are unchanged and source observation remains at 0 degrees.'
        training_note = 'Seed 42; two dynamically leased free GPUs; 5000 successful updates; same AP U-NO/MS noise/FM+Haar; sigma fixed at 1, with no new nonstationary branch.'
        gate_note = 'Relative to T2 on final validation: response improves at least 1%, native near-building error and transition gradient each improve at least 3%, paired interval upper bound is below zero, and overall error does not worsen by more than 1%. No next experiment launches automatically. Training loss cannot substitute for outcome comparison.'
    lines = [f'# {heading}', '', f"Updated: {state['time']}; status: {suite.get('status')}.", '',
        training_note,
        selection_note, '',
        '| Arm | Technical core | Phase | Formal step | Committed checkpoint | GPU |', '|---|---|---|---:|---:|---|']
    for arm, job in jobs.items():
        lines.append(f"| {arm} | {LABELS[arm]} | {job.get('phase')} | {job['progress'].get('step', 0)} | {job['committed_step']} | {job.get('gpus', [])} |")
    lines += ['', f'Free disk: {free:.1f} GiB. Alerts: ' + ('; '.join(alerts) if alerts else 'None'), '',
        gate_note, '',
        '## Current formal evaluation', '', '| Arm | Step | Split | Map RMSE dB | Unseen deltaRMSE dB |', '|---|---:|---|---:|---:|']
    latest = []
    for arm in arms:
        for split in ('val', 'test'):
            paths = sorted((root / 'jobs' / arm / 'evaluation').glob(f'step_*/{split}/summary.json'))
            if paths:
                path = paths[-1]; score = read(path); complete = read(path.parent / 'complete.json') if (path.parent / 'complete.json').exists() else {}
                latest.append({**complete, 'arm': arm, 'split': split, 'summary': score})
                lines.append(f"| {arm} | {complete.get('step')} | {split} | {score['map_rmse_db']:.4f} | {score['unseen_delta_rmse_db']:.4f} |")
    atomic(root / 'latest_results.json', latest)
    if transport:
        candidate_rows=[r for r in latest if r['arm']=='T3' and r['split']=='val']
        if candidate_rows:
            current=candidate_rows[-1];step=current['step']
            reference=root/'jobs/T2/evaluation'/f'step_{step:06d}'/'val/summary.json'
            if reference.exists():
                a,b=read(reference),current['summary']
                lines += ['',f'## Matched validation: T2 versus T3 at {step} steps','',
                    '| Metric | T2 reference | T3 | Relative decrease |','|---|---:|---:|---:|']
                for key in ('map_rmse_db','unseen_delta_rmse_db','native_near_rmse_db','shadow_transition_gradient_db_per_m'):
                    if a.get(key) is not None and b.get(key) is not None and a[key]>0:
                        lines.append(f'| {key} | {a[key]:.4f} | {b[key]:.4f} | {100*(1-b[key]/a[key]):.2f}% |')
    if cross_array:
        lines += ['', '## Regional metrics (height-map occlusion proxy, not path ground truth)', '',
            '| Arm | Step | Split | Native near RMSE dB | Transition RMSE dB | Transition gradient dB/m |',
            '|---|---:|---|---:|---:|---:|']
        for result in latest:
            score=result['summary']
            values=['NA' if score.get(k) is None else f'{score[k]:.4f}' for k in
                ('native_near_rmse_db','shadow_transition_rmse_db','shadow_transition_gradient_db_per_m')]
            lines.append('| '+' | '.join([result['arm'],str(result.get('step')),result['split'],*values])+' |')
    figures = sorted((root / 'figures').glob('*.png'))
    lines += ['', 'Early validation uses 16 fixed scenes and one draw; final evaluation uses 80 validation or 160 test scenes and four draws. Means from different sample sizes do not form a strict learning curve.', '',
        '## Visualizations', ''] + [f'![{path.stem}]({path})' for path in figures]
    lines += ['', '## Run safeguards', '', 'A persistent user systemd service survives SSH disconnects; dual-slot checkpoints every 250 updates restore model/Adam/scaler/scheduler; GPU locks, disk thresholds, stall detection, and bounded retries apply.',
        'The monitor updates every 30 seconds and figures every five minutes. Offline client push is not configured; alerts remain in this directory.',
        'This run studies power-map conditional response, not recovery of actual EES material, carrier phase, or ray paths.']
    atomic_text(root / 'REPORT.md', '\n'.join(lines))
    rows = ''.join(f"<tr><td>{arm}</td><td>{html.escape(job.get('phase', ''))}</td><td>{job['progress'].get('step', 0)}</td><td>{job['committed_step']}</td><td>{job.get('gpus', [])}</td></tr>" for arm, job in jobs.items())
    images = ''.join(f'<figure><img src="figures/{path.name}"><figcaption>{path.stem}</figcaption></figure>' for path in figures)
    atomic_text(root / 'index.html', f'<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Beam-boundary experiments</title><style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;background:#f5f6f7}}td,th{{padding:10px;border-bottom:1px solid #ddd}}figure{{background:white;padding:16px}}img{{max-width:100%}}</style><h1>{heading}</h1><p>{state["time"]} · {suite.get("status")}</p><p>{html.escape(selection_note)}</p><p>Seed 42; two GPUs per arm; 5000 updates; online weights. Power-feature phase is not carrier phase.</p><table><tr><th>Arm</th><th>Phase</th><th>Formal step</th><th>Saved step</th><th>GPU</th></tr>{rows}</table><p>Alerts: {html.escape("; ".join(alerts) or "None")}</p><p><a href="PROTOCOL.md">Frozen protocol</a> · <a href="REPORT.md">Detailed monitor</a> · <a href="latest_results.json">Evaluation results</a></p>{images}</html>')
    return suite.get('status')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--once', action='store_true')
    args = p.parse_args(); root = args.root.resolve(); renderer = None; rendered = 0
    presentation_cwd = Path(__file__).resolve().parents[1]
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    while True:
        try:
            status = tick(root)
            atomic(root / 'monitor_health.json', dict(status='ok', updated=time.time()))
            if renderer is not None and renderer.poll() is not None:
                atomic(root / 'render_job.json', dict(exit_code=renderer.returncode, updated=time.time())); renderer = None
            if renderer is None and time.time() - rendered >= 300:
                with (root / 'render.log').open('a') as log:
                    renderer = subprocess.Popen([sys.executable, '-m', 'beam_boundary.render', '--root', str(root)], cwd=presentation_cwd, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                rendered = time.time()
            if args.once or status == 'complete':
                if renderer is not None:
                    renderer.wait(); atomic(root / 'render_job.json', dict(exit_code=renderer.returncode, updated=time.time()))
                if status == 'complete':
                    result = subprocess.run([sys.executable, '-m', 'beam_boundary.render', '--root', str(root)], cwd=presentation_cwd, env=env)
                    atomic(root / 'render_job.json', dict(exit_code=result.returncode, updated=time.time()))
                tick(root); return
        except Exception:
            error = dict(updated=time.time(), traceback=traceback.format_exc())
            atomic(root / 'monitor_error.json', error)
            atomic(root / 'monitor_health.json', dict(status='failed', **error))
            atomic(root / 'alerts_latest.json', dict(updated=time.time(), alerts=['Monitor/report generation failed; monitor_error.json contains details']))
            if args.once: raise
        time.sleep(30)


if __name__ == '__main__': main()
