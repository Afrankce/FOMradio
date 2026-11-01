"""Post-hoc presentation of saved predictions; never changes training or selection."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from .common import CORE, atomic, atomic_text, read, sha, seed_for, verify
from .render import export, COLORS


def rows_at(root, arm, step, split):
    destination = root / 'jobs' / arm / 'evaluation' / f'step_{step:06d}' / split
    packs = [read(p) for p in sorted(destination.glob('rows_rank*.json'))]
    return destination, sum([p['maps'] for p in packs], []), sum([p['responses'] for p in packs], [])


def ci(values, seed):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(), n=len(values))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve(); verify(root)
    analysis = read(root / 'analysis/summary.json'); scores = analysis['summaries']
    output = root / 'stage_comparison'; output.mkdir(exist_ok=True)
    fingerprint = {}; coverage = {}; paired = {}; matched = {}; reference_keys = {}; gate_info = {}
    for split, n in [('val', 80), ('test', 160)]:
        coverage[split] = {}; paired[split] = {}
        for arm in CORE:
            dest, maps, responses = rows_at(root, arm, 5000, split)
            complete = read(dest / 'complete.json'); commit = read(root / 'jobs' / arm / 'committed.json')
            assert (complete['step'], complete['scenes'], complete['draws'], complete['nfe']) == (5000, n, 4, 8)
            assert complete['checkpoint_sha256'] == commit['sha256']
            keys = {(r['scene'], r['angle'], r['draw']) for r in maps}
            assert len(keys) == len(maps) == n * 8 * 4
            if split in reference_keys: assert keys == reference_keys[split]
            else: reference_keys[split] = keys
            measured = float(np.mean([r['rmse_db'] for r in maps]))
            assert abs(measured - scores[split][arm]['map_rmse_db']) < 1e-10
            delta = [r for r in responses if r['angle'] not in (-14., 0., 14.)]
            assert len(delta) == n * 5 * 4
            assert abs(np.mean([r['delta_rmse_db'] for r in delta]) - scores[split][arm]['unseen_delta_rmse_db']) < 1e-10
            fingerprint[str((dest / 'summary.json').relative_to(root))] = sha(dest / 'summary.json')
            coverage[split][arm] = dict(scenes=n, maps=len(maps), response_pairs=len(delta), exact_summary_recomputation=True)
            base = scores[split]['E0']['unseen_delta_by_scene']; mine = scores[split][arm]['unseen_delta_by_scene']
            paired[split][arm] = ci([mine[s] - base[s] for s in sorted(base)], seed_for(42, 'stage-baseline-ci', split))
    for arm in CORE:
        matched[arm] = {}; selected = None
        for step in (1000, 2500, 5000):
            dest, maps, response = rows_at(root, arm, step, 'val')
            if selected is None: selected = sorted({r['scene'] for r in maps})
            assert len(selected) == 16
            map_rows = [r for r in maps if r['scene'] in selected and r['draw'] == 0]
            delta_rows = [r for r in response if r['scene'] in selected and r['draw'] == 0 and r['angle'] not in (-14., 0., 14.)]
            assert len(map_rows) == 128 and len(delta_rows) == 80
            map_scene = {s: float(np.mean([r['rmse_db'] for r in map_rows if r['scene'] == s])) for s in selected}
            delta_scene = {s: float(np.mean([r['delta_rmse_db'] for r in delta_rows if r['scene'] == s])) for s in selected}
            matched[arm][step] = dict(map_rmse_db=float(np.mean(list(map_scene.values()))),
                delta_rmse_db=float(np.mean(list(delta_scene.values()))), map_by_scene=map_scene, delta_by_scene=delta_scene)
        if arm != 'E0':
            gate_info[arm] = {}
            for side in ('encoder', 'decoder'):
                path = root / 'jobs' / arm / 'evaluation/step_005000/val' / f'gate_{side}.npz'
                with np.load(path) as data:
                    gate_info[arm][side] = {key: dict(min=float(data[key].min()), max=float(data[key].max()), mean=float(data[key].mean()))
                        for key in ('source_gate', 'localized_source_rms', 'propagated_rms', 'residual_rms')}
    atomic(output / 'source_data.json', dict(scores=scores, paired=paired, matched=matched, gate_diagnostics=gate_info,
        coverage=coverage, input_sha256=fingerprint, plotting_source_sha256=sha(Path(__file__)),
        statistical_unit='Scene; 2000 percentile paired bootstrap resamples; single training seed; descriptive intervals, no multiplicity-adjusted significance claims',
        stage_population='Same 16 preregistered validation scenes and draw0 at all three checkpoints',
        selection='No changes to frozen validation phase-selection decision', updated=time.time()))
    # Primary question: do any added modules improve the pretrained-protocol backbone?
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    fig.subplots_adjust(left=.12, right=.98, bottom=.24, top=.84, wspace=.32)
    all_bounds = [v for split in paired.values() for arm, row in split.items() if arm != 'E0' for v in row['ci95']]
    lo, hi = min(-.02, min(all_bounds)) - .01, max(all_bounds) + .02
    for ax, split, title in zip(axes, ('val', 'test'), ('Validation: 80 scenes', 'Test: 160 scenes')):
        for i, arm in enumerate(CORE[1:]):
            row = paired[split][arm]; a, b = row['ci95']; center = row['mean']
            ax.errorbar(center, i, xerr=[[max(0, center-a)], [max(0, b-center)]], fmt='o', capsize=3, color=COLORS[i+1])
        ax.axvline(0, color='#777777', lw=.8, ls='--'); ax.set_xlim(lo, hi); ax.set_ylim(4.6, -.6)
        ax.set_xticks([0., .05, .10, .15])
        ax.set_title(title, fontsize=9); ax.set_yticks(range(5), CORE[1:]); ax.set_xlabel('Unseen deltaRMSE − E0 (dB)')
        ax.grid(axis='y', alpha=.15)
    export(fig, output / 'final_primary_comparison')
    # Matched, paired differences at checkpoints; spread belongs to scenes, not seeds.
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); fig.subplots_adjust(left=.12, bottom=.2, top=.8)
    stage_differences = {}
    for i, arm in enumerate(CORE[1:]):
        centers, lows, highs = [], [], []
        stage_differences[arm] = {}
        for step in (1000, 2500, 5000):
            mine = matched[arm][step]['delta_by_scene']; base = matched['E0'][step]['delta_by_scene']
            row = ci([mine[s] - base[s] for s in sorted(base)], seed_for(42, 'stage-paired', step))
            stage_differences[arm][step] = row; centers.append(row['mean']); lows.append(row['ci95'][0]); highs.append(row['ci95'][1])
        centers = np.array(centers)
        ax.errorbar([1000, 2500, 5000], centers, yerr=[centers-lows, np.array(highs)-centers],
                    color=COLORS[i+1], label=arm, marker='o', markersize=3, linewidth=1, capsize=2, alpha=.85)
    ax.axhline(0, color=COLORS[0], lw=.8, ls='--', label='E0')
    ax.set(xlabel='Successful optimizer updates', ylabel='Unseen deltaRMSE − E0 (dB)', xticks=[1000, 2500, 5000])
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=6)
    export(fig, output / 'matched_stage_comparison'); atomic(output / 'matched_stage_comparison.source.json', stage_differences)
    # Fixed mechanism comparators, not selected for visual quality or test ranking.
    scene = min(scores['val']['E0']['unseen_delta_by_scene'], key=lambda s: seed_for(42, 'visual-and-preview-scenes', 'val', s))
    samples = {}
    for arm in CORE:
        with np.load(root / 'jobs' / arm / 'evaluation/step_005000/val/predictions' / f'{scene}.npz') as data:
            samples[arm] = {key: data[key] for key in data.files}
        assert np.array_equal(samples[arm]['source_hashes'], samples['E0']['source_hashes'])
        assert np.array_equal(samples[arm]['truth_db'], samples['E0']['truth_db'])
    reference = samples['E0']; angles = list(reference['angles']); zero, seven = angles.index(0.), angles.index(7.)
    # Show all six arms and the target in a two-row wide internal comparison plate.
    fig, axes = plt.subplots(2, 7, figsize=(14.4, 4.6))
    fig.subplots_adjust(left=.055, right=.935, bottom=.08, top=.90, wspace=.08, hspace=.12)
    fields = [reference['truth_db']] + [samples[a]['prediction_db'].mean(0) for a in CORE]
    for col, (title, fields_for_arm) in enumerate(zip(('Target',) + CORE, fields)):
        axes[0, col].set_title(title, fontsize=10)
        power = np.where(reference['valid'][zero], fields_for_arm[zero], np.nan)
        mask = reference['valid'][seven] & reference['valid'][zero]
        delta = np.where(mask, fields_for_arm[seven] - fields_for_arm[zero], np.nan)
        a = axes[0, col].imshow(power, vmin=-180, vmax=-40, cmap='viridis')
        b = axes[1, col].imshow(delta, vmin=-40, vmax=40, cmap='RdBu_r')
        for row in range(2): axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel('Power at 0°'); axes[1, 0].set_ylabel('Response: +7° − 0°')
    colorbars = []
    for row, image in enumerate((a, b)):
        pos = axes[row, 0].get_position(); cax = fig.add_axes([.945, pos.y0, .009, pos.height])
        bar = fig.colorbar(image, cax=cax, extend='both'); bar.set_label('dB'); colorbars.append(cax)
    export(fig, output / 'fixed_scene_all_arms', colorbars)
    atomic(output / 'fixed_scene_all_arms.source.json', dict(scene=scene, split='val', selection='Pre-training fixed scene hash',
        display='Mean of four paired source draws. Scalar-map response, not ray-path ground truth.',
        spatial='Full 256 model grid; no crop, no per-panel rescaling; invalid regions white',
        limits='Power [-180,-40] dB, response [-40,40] dB, color-scale clipping only',
        raw_shapes={a: list(s['prediction_db'].shape) for a,s in samples.items()}))
    # Detailed scalar metrics remain available without cherry-picking favorable ones.
    keys = ['map_rmse_db', 'unseen_delta_rmse_db', 'native_rmse_db', 'near_rmse_db', 'open_rmse_db',
            'gradient_rmse_db_per_m', 'haar1_rmse_db', 'haar2_rmse_db', 'haar3_rmse_db', 'haar4_rmse_db', 'orientation_mae_deg']
    with (output / 'final_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split', 'arm'] + keys); writer.writeheader()
        for split in ('val', 'test'):
            for arm in CORE: writer.writerow(dict(split=split, arm=arm, **{k: scores[split][arm][k] for k in keys}))
    labels = dict(E0='NoDrop AP', E1='Spatially averaged source gating', E2='Beam-boundary Fourier', E3='Geometry-only source gating', E4='Parameter-matched local convolution', E5='Without extra amplitude features')
    lines = ['# Single-seed beam-boundary stage comparison', '', 'All six core trainings and validation/test evaluations completed; P-A/P-AP were skipped under the preregistered rule.', '',
        'Protocol: seed 42; 5000 updates; online weights; Euler8; 80 validation and 160 test scenes; eight angles x four source draws per scene.',
        'The primary metric is angle-response error relative to 0 degrees; lower is better throughout. Near-building regions derive from input heights, not reflection-path labels.', '',
        '## Final results (ranked by test unseen-angle response error)', '',
        '| Arm | Technical core | Val delta RMSE | Test delta RMSE | Test map RMSE | Test native128 RMSE |', '|---|---|---:|---:|---:|---:|']
    for arm in sorted(CORE, key=lambda a: scores['test'][a]['unseen_delta_rmse_db']):
        s=scores['test'][arm]; lines.append(f"| {arm} | {labels[arm]} | {scores['val'][arm]['unseen_delta_rmse_db']:.4f} | {s['unseen_delta_rmse_db']:.4f} | {s['map_rmse_db']:.4f} | {s['native_rmse_db']:.4f} |")
    lines += ['', '## Matched-scene, matched-draw training stages', '', 'Only the original 16 fixed validation scenes and draw 0 are used, permitting stage comparisons. Preview means over 16 x 1 must not be joined directly to final means over 80 x 4.', '',
        '| Arm | 1000-step delta RMSE | 2500-step delta RMSE | 5000-step delta RMSE |', '|---|---:|---:|---:|']
    for arm in CORE: lines.append('| '+arm+' | '+' | '.join(f"{matched[arm][s]['delta_rmse_db']:.4f}" for s in (1000,2500,5000))+' |')
    lines += ['', '## Evidence-based stage interpretation', '',
        '- E0 remains the best point estimate on the primary validation and test metric; added E2 complexity has not yielded an overall benefit.',
        '- E2 versus E0: validation difference +0.0814 dB, scene-bootstrap 95% interval [+0.0491,+0.1144]; test difference +0.0148 dB, interval [-0.0107,+0.0395].',
        '- E2 versus E1: a small test advantage, but the validation interval crosses zero; stable benefit from spatial gating is not established.',
        '- E2 versus E3: adding Beam to source gating shows no further benefit; this does not imply that the backbone Beam condition is useless.',
        '- E2 versus E4: test response exceeds this local comparator, but the validation difference is unclear and says nothing about all local CNNs.',
        '- E2 versus E5: the extra amplitude transform has little benefit, with both intervals crossing zero; further amplitude/phase modules are not prioritized.',
        '- Improvements in gradient, Haar, or orientation metrics disagree with native-grid and near-building errors; one better detail metric does not establish recovered reflection physics.',
        '- Stored gates and residuals are nonzero in a fixed diagnostic scene, so the module is not entirely inactive; activation is not evidence of an effective propagation mechanism.', '',
        '## Suggested priorities (no new training launched)', '',
        '1. Retain E0 as baseline; use trained checkpoints for controlled branch-zeroing and shuffled-source-gate diagnostics to measure module contribution.',
        '2. Check the native 128 grid and near-wall regions: the 256 target is interpolated, so boundary errors also reflect interpolation and invalid masks, not only FNO.',
        '3. If those diagnostics support an architecture change, prioritize direction/visibility-constrained boundary interactions over untested EES or RF-phase assumptions.', '',
        '## Visualizations', '',
        f'![Paired primary metric comparison]({output / "final_primary_comparison.png"})', '',
        f'![Matched-stage comparison]({output / "matched_stage_comparison.png"})', '',
        f'![Fixed-scene all-arm comparison]({output / "fixed_scene_all_arms.png"})', '',
        'Intervals describe scene-sampling variation, not stability over training seeds; no multiplicity-adjusted significance claim is made.',
        'The monitor page previously stopped refreshing due to a duplicate split field. This fix affected presentation only; frozen training/evaluation code and numerical results were unchanged.']
    atomic_text(output / 'REPORT.md', '\n'.join(lines))
    atomic(output / 'figure_contract.json', dict(backend='Python', scientific_question='Do the added conditional residuals improve held-out steering response?',
        evidence='Full validation/test paired scene differences; matched checkpoint progress; fixed-scene qualitative control',
        widths='7.2 inch quantitative figures; 14.4 inch internal all-arm image plate for inspection, not a journal submission layout',
        source_validation='Exact summary recomputation and equal sample-key coverage; immutable frozen code verified',
        exclusions='Only qualitative image and stage curve use the preregistered fixed subset; final table uses all scenes',
        scene_bootstrap='2000 paired resamples, percentile95%, single seed; no adjusted significance claim'))
    print(output / 'REPORT.md', flush=True)


if __name__ == '__main__': main()
