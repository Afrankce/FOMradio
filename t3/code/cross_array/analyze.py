"""Validation-frozen progression decision; paired scenes, one training seed."""
import argparse
from pathlib import Path
import time
import numpy as np
from beam_boundary.common import atomic, atomic_text, read, seed_for, sha
from .geometry import REGIONAL_KEYS


def paired(a, b, split, key):
    if set(a) != set(b): raise RuntimeError(f'Paired scene coverage mismatch: {key}')
    names = sorted(a)
    if not names: return dict(n_scenes=0, ci95_scene_bootstrap=None, relative_improvement=None)
    left = np.array([a[k] for k in names]); right = np.array([b[k] for k in names])
    if not np.isfinite(left).all() or not np.isfinite(right).all(): raise ValueError('Nonfinite scores')
    difference = left-right
    rng = np.random.default_rng(seed_for(42, 'cross-array-bootstrap', split, key))
    boot = difference[rng.integers(0, len(names), size=(2000, len(names)))].mean(1)
    return dict(mean_difference=float(difference.mean()), mean_delta_rmse_difference_db=float(difference.mean()),
        ci95_scene_bootstrap=np.quantile(boot, [.025, .975]).tolist(), n_scenes=len(names),
        relative_improvement=float(1-left.mean()/right.mean()) if right.mean()>0 else None,
        baseline_mean=float(right.mean()), candidate_mean=float(left.mean()), training_seeds=1,
        per_scene=dict(zip(names, difference.tolist())))


def decide(scores, effects, gate, baseline='T0', candidate='T2'):
    thresholds = dict(unseen_delta_rmse_db=gate['unseen_delta_improvement'],
        native_near_rmse_db=gate['native_near_improvement'],
        shadow_transition_gradient_db_per_m=gate['shadow_transition_gradient_improvement'])
    checks = {}; inconclusive = False
    for key, threshold in thresholds.items():
        e = effects[key]; required = 80 if key=='unseen_delta_rmse_db' else gate['min_region_scenes']
        covered = e['n_scenes'] >= required and e['relative_improvement'] is not None
        checks[key] = dict(coverage_pass=covered, minimum_scenes=required, improvement_threshold=threshold,
            pass_gate=bool(covered and e['relative_improvement']>=threshold and e['ci95_scene_bootstrap'][1]<0), **e)
        inconclusive |= not covered
    checks['global_guardrails'] = {k: scores[candidate][k] <= gate['map_guardrail']*scores[baseline][k]
        for k in ('map_rmse_db','native_rmse_db')}
    passed = all(checks[k]['pass_gate'] for k in thresholds) and all(checks['global_guardrails'].values())
    return dict(decision='inconclusive' if inconclusive else 'supported' if passed else 'not_supported', checks=checks,
        validation_only=True, next_action='Review next isolated ablation' if passed else 'Do not advance; inspect stage evidence',
        automatic_next_stage_launch=False, training_seeds=1, baseline_arm=baseline, candidate_arm=candidate)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--validation-only',action='store_true')
    args=p.parse_args();root=args.root.resolve();cfg=read(root/'config.json')
    baseline,candidate=cfg.get('baseline_arm','T0'),cfg.get('candidate_arm','T2')
    summaries={};comparisons={};regional={};validation_hashes={}
    for split in (('val',) if args.validation_only else ('val','test')):
        scores={}
        for arm in (baseline,candidate):
            folder=root/'jobs'/arm/'evaluation'/f"step_{cfg['total_steps']:06d}"/split
            complete=read(folder/'complete.json'); s=read(folder/'summary.json')
            count=80 if split=='val' else 160
            assert complete['step']==cfg['total_steps'] and complete['draws']==cfg['eval_draws'] and complete['nfe']==cfg['eval_nfe']
            assert complete['scenes']==s['scenes']==count and s['maps']==count*8*cfg['eval_draws']
            assert len(s['unseen_delta_by_scene'])==count
            scores[arm]=s
            if split=='val': validation_hashes[arm]=sha(folder/'summary.json')
        effects={'unseen_delta_rmse_db':paired(scores[candidate]['unseen_delta_by_scene'],scores[baseline]['unseen_delta_by_scene'],split,'unseen')}
        for key in REGIONAL_KEYS:
            effects[key]=paired(scores[candidate][key+'_by_scene'],scores[baseline][key+'_by_scene'],split,key)
        summaries[split]=scores;comparisons[split]={f'{candidate}-minus-{baseline}':effects['unseen_delta_rmse_db']};regional[split]=effects
    gate_path=root/'progression_gate.json'
    if gate_path.exists():
        decision=read(gate_path)
        if decision['validation_sha256']!=validation_hashes: raise RuntimeError('Frozen validation decision input changed')
    else:
        if not args.validation_only: raise RuntimeError('Freeze validation decision before test analysis')
        decision=decide(summaries['val'],regional['val'],cfg['progression_gate'],baseline,candidate)
        decision.update(validation_sha256=validation_hashes,updated=time.time());atomic(gate_path,decision)
    atomic(root/'analysis/summary.json',dict(summaries=summaries,comparisons=comparisons,regional_comparisons=regional,
        progression=decision,interval_interpretation='2000 paired scene resamples; no training-seed stability claim'))
    source_note = ('T3 starts from the fixed real Y8 observation plus the original MS noise; '
                   'T2 uses Y8 only as a condition. The network, loss, noise scale, and budget are unchanged.'
        if cfg.get('experiment_kind')=='cross_array_transport' else 'The original noise source and FM loss are unchanged.')
    lines=[f'# {baseline} / {candidate}: single-seed cross-array comparison','',
        '8x8 and 32x32 denote antenna arrays; the source observation is fixed at 0 degrees '
        'and the target uses a 32x32 array. '+source_note,'',
        '| Split | Arm | Map RMSE dB | Unseen ΔRMSE dB | Native near RMSE dB | Transition gradient dB/m |',
        '|---|---|---:|---:|---:|---:|']
    def fmt(value): return 'NA' if value is None else f'{value:.4f}'
    for split,scores in summaries.items():
        for arm,s in scores.items():
            values=[fmt(s[k]) for k in ('map_rmse_db','unseen_delta_rmse_db','native_near_rmse_db','shadow_transition_gradient_db_per_m')]
            lines.append('| '+ ' | '.join([split,arm,*values])+' |')
    lines += ['',f"Prespecified validation gate: {decision['decision']}; next action: {decision['next_action']}.",
        'Do not launch another experiment or a new nonstationary fusion automatically. The test split does not inform this decision.',
        'Shadow and transition regions are line-of-sight proxies from the height map, not ground-truth ray, reflection, or diffraction labels.',
        'Complete real Y8 is an additional observation; findings do not extend to absent, sparse, or predicted Y8 input.',
        'Paired-scene bootstrap captures scene-sampling variation only; this run uses one training seed.']
    if cfg.get('experiment_kind')=='cross_array_transport':
        lines += ['', '## Direct source-Y8 copy without a model', '',
            'Deterministic reference scored once per scene and angle; the target-valid mask is unchanged. '
            'Missing source cells use the same normalized zero fill as T3.', '',
            '| Split | Map RMSE dB | Unseen ΔRMSE dB | Native near RMSE dB | Transition gradient dB/m |',
            '|---|---:|---:|---:|---:|']
        for split in ('val','test'):
            path=root/'source_copy'/split/'summary.json'
            if path.exists():
                s=read(path);values=[fmt(s[k]) for k in ('map_rmse_db','unseen_delta_rmse_db','native_near_rmse_db','shadow_transition_gradient_db_per_m')]
                lines.append('| '+' | '.join([split,*values])+' |')
    atomic_text(root/'analysis/RESULTS.md','\n'.join(lines))


if __name__=='__main__': main()
