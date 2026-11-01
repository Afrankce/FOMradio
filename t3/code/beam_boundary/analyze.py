"""Final validation/test comparison with paired scene bootstrap, not seed CI."""
import argparse
from pathlib import Path
import numpy as np
from .common import CORE, PHASE, atomic, atomic_text, read, seed_for


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); root = p.parse_args().root.resolve()
    cfg = read(root / 'config.json'); summaries = {}; comparisons = {}
    for split in ('val', 'test'):
        scores = {}
        for arm in CORE + PHASE:
            path = root / 'jobs' / arm / 'evaluation' / f"step_{cfg['total_steps']:06d}" / split / 'summary.json'
            if path.exists(): scores[arm] = read(path)
        summaries[split] = scores; comparisons[split] = {}
        pairs = cfg.get('comparison_pairs', [('E2', 'E0'), ('E2', 'E1'), ('E2', 'E3'), ('E2', 'E4'), ('E2', 'E5'), ('P-AP', 'P-A')])
        for left, right in pairs:
            if left not in scores or right not in scores: continue
            a, b = scores[left]['unseen_delta_by_scene'], scores[right]['unseen_delta_by_scene']
            if set(a) != set(b): raise RuntimeError('Paired scene coverage mismatch')
            differences = np.array([a[k] - b[k] for k in sorted(a)])
            rng = np.random.default_rng(seed_for(42, 'scene-bootstrap', split))
            boot = differences[rng.integers(0, len(differences), size=(2000, len(differences)))].mean(1)
            comparisons[split][f'{left}-minus-{right}'] = dict(mean_delta_rmse_difference_db=float(differences.mean()),
                ci95_scene_bootstrap=np.quantile(boot, [.025, .975]).tolist(), n_scenes=len(differences),
                training_seeds=1, per_scene=dict(zip(sorted(a), differences.tolist())))
    atomic(root / 'analysis/summary.json', dict(summaries=summaries, comparisons=comparisons,
        interval_interpretation='Paired resampling of held-out scenes; does not quantify training-seed stability'))
    lines = ['# Single-seed beam-boundary comparison', '', 'Only completed online final5000 / Euler8 results are reported.', '',
        '| Split | Arm | Map RMSE dB | Unseen deltaRMSE dB | Native128 RMSE dB |', '|---|---|---:|---:|---:|']
    for split, scores in summaries.items():
        for arm, s in sorted(scores.items(), key=lambda item: item[1]['unseen_delta_rmse_db']):
            lines.append(f"| {split} | {arm} | {s['map_rmse_db']:.4f} | {s['unseen_delta_rmse_db']:.4f} | {s['native_rmse_db']:.4f} |")
    lines += ['', 'See phase_selection.json for phase-base selection; bases were ranked on validation data.',
              'The supplement was proposed after observing the previous round and is exploratory.' if cfg.get('experiment_kind') == 'phase_supplement' else 'The original conditional gate was unchanged.',
              'Scene-bootstrap intervals describe scene-sampling variation, not stability across training seeds.',
              'Local directions are power-map morphology; these results alone do not prove recovery of reflection paths or EES scattering.']
    atomic_text(root / 'analysis/RESULTS.md', '\n'.join(lines))


if __name__ == '__main__': main()
