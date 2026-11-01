"""Actual-data monitoring figures; fixed hash scenes, no inferred results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import CORE, PHASE, atomic, read, seed_for, sha
from rf_baseline.metrics import gradients, haar, orientation

SKILL_SCRIPTS = Path('/home/ka/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'axes.linewidth': .7})
COLORS = ['#6f747b', '#9b8ca6', '#277c91', '#bc865b', '#789b83', '#7486bd', '#ac7691', '#235367']


def export(fig, path, exclude_axes=()):
    fig.canvas.draw()
    panels = [ax for ax in fig.axes if ax not in exclude_axes]
    identifiers = [f'p{i}' for i in range(len(panels))]
    groups = {}
    for identifier, ax in zip(identifiers, panels):
        groups.setdefault(round(ax.get_position().y0, 4), []).append(identifier)
    rows = [group for group in groups.values() if len(group) > 1]
    require_matplotlib_panel_alignment(fig, json_out=str(path) + '.alignment.json',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, axes=panels, panel_ids=identifiers,
        row_groups=rows, exclude_axes=list(exclude_axes), strict=True)
    fig.savefig(f'{path}.svg', bbox_inches='tight')
    fig.savefig(f'{path}.pdf', bbox_inches='tight')
    fig.savefig(f'{path}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    for script, arguments, name in [
        ('audit_pdf_text.py', ['--min-pt', '5', '--json'], 'text-audit'),
        ('audit_figure_collisions.py', ['--json-out', str(path) + '.collision-audit.json'], 'collision-log')]:
        qa_python = '/home/ka/CKM/radio-unet/work/rf-figure-qa-env/bin/python' if script == 'audit_figure_collisions.py' else sys.executable
        result = subprocess.run([qa_python, str(SKILL_SCRIPTS / script), str(path) + '.pdf', *arguments], capture_output=True, text=True)
        atomic(str(path) + f'.{name}.json', dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
        if result.returncode: raise RuntimeError(f'Figure QA failed: {path.name}: {script}: {result.stdout[-2000:]}')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--arm', choices=CORE + PHASE + ('T0', 'T2', 'T3'))
    args = p.parse_args(); root = args.root.resolve()
    folder = root / ('preflight_figures' if args.smoke else 'figures'); folder.mkdir(exist_ok=True); data = {}; cfg = read(root / 'config.json')
    arms = tuple(cfg.get('display_arms', CORE + PHASE))
    colors = dict(zip(CORE + PHASE, COLORS)); colors.update(T0='#6f747b', T2='#277c91', T3='#277c91')
    transport = cfg.get('experiment_kind') == 'cross_array_transport'
    if transport: colors['T2'] = '#6f747b'
    arm_colors = [(arm, colors[arm]) for arm in arms]
    cross_array = cfg.get('experiment_kind') in ('cross_array_observation','cross_array_transport')
    atomic(folder / 'presentation_provenance.json', dict(source=str(Path(__file__).resolve()),
        source_sha256=sha(Path(__file__)), note='Presentation layer may be repaired independently; frozen training source is unchanged'))
    for arm in arms:
        path = root / 'jobs' / arm / 'history.json'
        if path.exists(): data[arm] = read(path)
    if data:
        fig, ax = plt.subplots(figsize=(7.2, 3.8)); fig.subplots_adjust(top=.78, bottom=.16)
        for arm, color in arm_colors:
            rows = data.get(arm, [])
            if rows: ax.plot([r['step'] for r in rows], [r['loss'] for r in rows], label=arm, color=color, lw=1.3)
        ax.set(xlabel='Successful optimizer updates', ylabel='Velocity loss (different targets)' if transport else 'FM + Haar velocity loss')
        ax.set_yscale('log'); ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=8)
        ax.grid(alpha=.15); export(fig, folder / 'training_loss')
        atomic(folder / 'training_loss.source.json', data)
        fig, ax = plt.subplots(figsize=(7.2, 3.8)); fig.subplots_adjust(top=.78, bottom=.16)
        for arm, color in arm_colors:
            rows = data.get(arm, [])
            if rows: ax.plot([r['step'] for r in rows], [r['seconds'] for r in rows], label=arm, color=color, lw=1.1)
        ax.set(xlabel='Successful optimizer updates', ylabel='Seconds per update (2 GPUs)')
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=8); export(fig, folder / 'step_time')
    # Honest pre-training visualization: inputs only, not a generated result.
    if not (folder / 'fixed_inputs.source.json').exists():
        records = read(root / 'data/val_records.json')
        scene = min({r['scene_id'] for r in records}, key=lambda s: seed_for(42, 'visual-and-preview-scenes', 'val', s))
        i = next(i for i, r in enumerate(records) if r['scene_id'] == scene and r['steering_deg'] == 0)
        c = np.load(root / 'data/val_condition.npy', mmap_mode='r')[i]
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0)); fig.subplots_adjust(wspace=.3, top=.83, bottom=.1)
        fields = [c[1] * 445, c[2] * 300 - 300, np.logaddexp(0, (c[2] * 300 - 220) * np.log(10) / 20) / 10]
        labels = ['Height (m)', 'Beam (dB)', 'Relative amplitude feature']
        if cross_array:
            fields[2] = np.where(c[5]>0, c[4]*300-300, np.nan)
            labels = ['Height (m)', 'Target 32×32 Beam (dB)', 'Observed 8×8 Y, 0° (dB)']
        colorbars = []
        for ax, field, label in zip(axes, fields, labels):
            image = ax.imshow(field, origin='upper', cmap='viridis', extent=(0, 1280, 1280, 0)); ax.set_title(label, fontsize=8)
            ax.set_xticks([0, 640, 1280]); ax.set_yticks([0, 640, 1280]); ax.set_xlabel('x (m)')
            bar = fig.colorbar(image, ax=ax, orientation='horizontal', fraction=.06, pad=.27); colorbars.append(bar.ax)
        axes[0].set_ylabel('y (m)'); export(fig, folder / 'fixed_inputs', colorbars)
        atomic(folder / 'fixed_inputs.source.json', dict(scene=scene, angle=0, split='val', selection='predefined scene hash',
            claim='Available input representations only; no generated results or performance claim'))
        if cross_array:
            with np.load(root / 'data/geometry' / f'{scene}.npz') as regions:
                fields = [regions[k].astype(float) for k in ('near','shadow','transition')]
            fig, axes = plt.subplots(1,3,figsize=(7.2,2.8));fig.subplots_adjust(wspace=.3,top=.86,bottom=.16)
            for ax,field,title in zip(axes,fields,['Near building ≤20 m','Straight-line shadow proxy','Transition proxy ≤20 m']):
                ax.imshow(field,cmap='Greys',vmin=0,vmax=1,extent=(0,1280,1280,0))
                ax.set_title(title,fontsize=8);ax.set_xticks([0,640,1280]);ax.set_yticks([0,640,1280]);ax.set_xlabel('x (m)')
            axes[0].set_ylabel('y (m)');export(fig,folder/'geometry_proxies')
            atomic(folder/'geometry_proxies.source.json',dict(scene=scene,source=str(root/'data/geometry'/f'{scene}.npz'),
                claim='Geometry-only scoring regions; black=selected; not reflection ground truth',native_spacing_m=10))
    for arm in arms:
        if args.arm and args.arm != arm: continue
        paths = sorted((root / 'jobs' / arm / 'evaluation').glob('smoke/val/complete.json' if args.smoke else 'step_*/val/complete.json'))
        if not paths: continue
        dest = paths[-1].parent; complete = read(paths[-1]); marker = folder / f'{arm}_visualized.json'
        if marker.exists() and read(marker).get('checkpoint_sha256') == complete['checkpoint_sha256']: continue
        predictions = sorted((dest / 'predictions').glob('*.npz'), key=lambda path: seed_for(42, 'visual-and-preview-scenes', 'val', path.stem))
        if not predictions: continue
        sample = np.load(predictions[0]); angles = sample['angles']; zero = list(angles).index(0.); seven = list(angles).index(7.)
        truth = sample['truth_db']; pred = sample['prediction_db'].mean(0); valid = sample['valid']
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.8)); fig.subplots_adjust(hspace=.8, wspace=.3, top=.93, bottom=.08)
        fields = [sample['height'] * 445, sample['beam_normalized'][zero] * 300 - 300, truth[zero], pred[zero],
                  truth[seven] - truth[zero], pred[seven] - pred[zero]]
        titles = ['Height (m)', 'Beam 0° (dB)', 'Target 0° (dB)', 'Generated mean 0° (dB)', 'Target Δ(+7°, 0°)', 'Generated mean Δ(+7°, 0°)']
        colorbars = []
        for j, (ax, field, title) in enumerate(zip(axes.flat, fields, titles)):
            support = valid[zero] & valid[seven] if j >= 4 else valid[zero]
            shown = np.where(support, field, np.nan) if j >= 2 else field
            limits = dict(vmin=-40, vmax=40, cmap='RdBu_r') if j >= 4 else dict(vmin=-180, vmax=-40, cmap='viridis') if j in (2, 3) else dict(cmap='viridis')
            im = ax.imshow(shown, origin='upper', **limits); ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            colorbars.append(fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.055, pad=.10, extend='both' if j >= 2 else 'neither').ax)
        export(fig, folder / f'{arm}_fixed_scene', colorbars)
        # Diagnostics use the same fixed scene, not an independently selected example.
        tgx, tgy, ts = gradients(truth[seven], valid[seven], spacing=5.)
        pgx, pgy, ps = gradients(pred[seven], valid[seven], spacing=5.)
        ta, tc, _, tos = orientation(truth[seven], valid[seven], spacing=5.)
        pa, pc, _, pos = orientation(pred[seven], valid[seven], spacing=5.)
        safe = tos & pos & (tc >= .2)
        grad_max = max(1e-6, float(np.quantile(np.hypot(tgx, tgy)[ts], .99))) if ts.any() else 1.
        angle_error = .5 * abs(np.angle(np.exp(2j * (pa - ta)))) * 180 / np.pi
        diagnostic_fields = [np.where(ts, np.hypot(tgx, tgy), np.nan), np.where(ps, np.hypot(pgx, pgy), np.nan),
            np.where(ts & ps, np.hypot(pgx - tgx, pgy - tgy), np.nan),
            np.where(safe, ta * 180 / np.pi, np.nan), np.where(safe, pa * 180 / np.pi, np.nan), np.where(safe, angle_error, np.nan)]
        titles = ['Target gradient (dB/m)', 'Generated gradient (dB/m)', 'Gradient vector error',
                  'Target morphology axis (°)', 'Generated morphology axis (°)', 'Axial orientation error (°)']
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.8)); fig.subplots_adjust(hspace=.8, wspace=.3, top=.93, bottom=.08)
        colorbars = []
        for j, (ax, field, title) in enumerate(zip(axes.flat, diagnostic_fields, titles)):
            im = ax.imshow(field, vmin=0, vmax=grad_max if j < 3 else 180 if j < 5 else 90,
                           cmap='twilight' if j in (3, 4) else 'viridis')
            ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            colorbars.append(fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.055, pad=.10).ax)
        export(fig, folder / f'{arm}_morphology', colorbars)
        truth_haar, pred_haar = haar(truth[seven], valid[seven]), haar(pred[seven], valid[seven])
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6)); fig.subplots_adjust(hspace=.85, wspace=.3, top=.93, bottom=.08)
        colorbars = []
        for col, ((level, target_detail, support), (_, pred_detail, _)) in enumerate(zip(truth_haar, pred_haar)):
            target_rms = np.sqrt(np.mean(target_detail**2, axis=0)); pred_rms = np.sqrt(np.mean(pred_detail**2, axis=0))
            scale = max(1e-6, float(np.quantile(target_rms[support], .99))) if support.any() else 1.
            for row, field in enumerate((target_rms, pred_rms)):
                ax = axes[row, col]; im = ax.imshow(np.where(support, field, np.nan), vmin=0, vmax=scale, cmap='viridis')
                ax.set_title(('Target' if row == 0 else 'Generated') + f' Haar {level}', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                colorbars.append(fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.055, pad=.10).ax)
        export(fig, folder / f'{arm}_haar_details', colorbars)
        for side in ('encoder', 'decoder'):
            path = dest / f'gate_{side}.npz'
            if not path.exists(): continue
            gate = np.load(path); keys = ['source_gate', 'pi', 'alpha', 'localized_source_rms', 'propagated_rms', 'residual_rms']
            fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.8)); fig.subplots_adjust(hspace=.8, wspace=.3, top=.93, bottom=.08)
            colorbars = []
            for ax, key in zip(axes.flat, keys):
                value = gate[key][0] if key in ('source_gate', 'pi', 'alpha') else gate[key].squeeze()
                limits = dict(vmin=0, vmax=1) if key in ('source_gate', 'pi', 'alpha') else dict(vmin=0, vmax=max(1e-8, float(np.max(value))))
                image = ax.imshow(value, cmap='viridis', **limits); ax.set_title(key.replace('_', ' '), fontsize=8)
                ax.set_xticks([]); ax.set_yticks([]); colorbars.append(fig.colorbar(image, ax=ax, orientation='horizontal', fraction=.055, pad=.10).ax)
            export(fig, folder / f'{arm}_{side}_gates', colorbars)
            if 'beam_spectral_gate_rotation' in gate:
                spectral_keys = ['beam_log_magnitude', 'hidden_log_magnitude', 'relative_phase_real',
                                 'relative_phase_imag', 'gate_magnitude', 'gate_rotation']
                labels = ['Beam log-magnitude', 'Hidden log-magnitude', 'Phase input: real',
                          'Phase input: imaginary', 'Output spectral gain', 'Output rotation (rad)']
                fields = [gate[f'beam_spectral_{key}'][0] for key in spectral_keys]
                max_log = max(1e-6, float(max(fields[0].max(), fields[1].max())))
                limits = [(0, max_log), (0, max_log), (-1, 1), (-1, 1), (.5, 1.5), (-np.pi/4, np.pi/4)]
                fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.8))
                fig.subplots_adjust(hspace=.85, wspace=.35, top=.93, bottom=.1); colorbars = []
                for j, (ax, field, title, (lo, hi)) in enumerate(zip(axes.flat, fields, labels, limits)):
                    image = ax.imshow(field, origin='lower', aspect='auto', vmin=lo, vmax=hi,
                        extent=(-.5, field.shape[1]-.5, -(field.shape[0]//2)-.5, field.shape[0]//2+.5),
                        cmap='RdBu_r' if j in (2, 3) else 'twilight' if j == 5 else 'viridis')
                    ax.set_title(title, fontsize=8); ax.set_xlabel('kx mode'); ax.set_ylabel('ky mode')
                    colorbars.append(fig.colorbar(image, ax=ax, orientation='horizontal', fraction=.055, pad=.32).ax)
                export(fig, folder / f'{arm}_{side}_beam_spectrum', colorbars)
                atomic(folder / f'{arm}_{side}_beam_spectrum.source.json', dict(
                    source=str(path), source_sha256=sha(path), channel=0, scene=str(gate['scene']),
                    fm_time=float(gate['fm_time']), angle=float(gate['angle']),
                    phase_semantics='RMS-normalized, low-energy-attenuated spatial cross-phase inputs; not RF phase',
                    contrast='P-A phase input slots are zero; both arms share gain/rotation output capability',
                    limits=limits, display='Same fixed scene/time; first latent channel; not a physical angular scattering spectrum'))
        atomic(marker, dict(**complete, scene=predictions[0].stem, display='Mean of saved source draws; metrics use individual draws',
            gate_display='First expert pi/alpha; input-side gate and propagated response are separate diagnostics',
            gate_precision='Frozen capture RMS is evaluated in its activation dtype; tiny FP16 RMS can underflow. Zero does not prove an inactive branch.',
            color_limits='Target/generated power [-180,-40] dB; both deltas [-40,40] dB, visual clipping only',
            morphology='Angle +7 degrees, derivative spacing5m, axial directions are scalar power morphology, not rays',
            diagnostic_color_limits='Gradient and Haar shared target99th-percentile bounds; orientation [0,180] and axial error [0,90] degrees'))
    scores = {}
    for arm in arms:
        path = root / 'jobs' / arm / 'evaluation' / f"step_{cfg['total_steps']:06d}" / 'val/summary.json'
        if path.exists(): scores[arm] = read(path)
    if scores:
        fig, ax = plt.subplots(figsize=(7.2, 4)); fig.subplots_adjust(top=.82, bottom=.16)
        for arm, color in arm_colors:
            if arm in scores: ax.scatter(scores[arm]['map_rmse_db'], scores[arm]['unseen_delta_rmse_db'], color=color, label=arm, s=35)
        ax.set(xlabel='Validation map RMSE (dB)', ylabel='Validation unseen-angle deltaRMSE (dB)')
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=8); export(fig, folder / 'validation_tradeoff')
        atomic(folder / 'validation_tradeoff.source.json', scores)
    analysis_path = root / 'analysis/summary.json'
    if analysis_path.exists():
        analysis = read(analysis_path)
        for split, comparisons in analysis['comparisons'].items():
            if not comparisons: continue
            fig, ax = plt.subplots(figsize=(7.2, 3.8)); fig.subplots_adjust(left=.24, bottom=.2)
            for i, (name, values) in enumerate(comparisons.items()):
                lo, hi = values['ci95_scene_bootstrap']; center = values['mean_delta_rmse_difference_db']
                ax.plot([lo, hi], [i, i], color='#277c91', lw=2); ax.scatter([center], [i], color='#277c91', s=28, zorder=3)
            ax.axvline(0, color='#888888', ls='--', lw=.8)
            ax.set_yticks(range(len(comparisons)), [name.replace('-minus-', ' − ') for name in comparisons])
            ax.set_xlabel('Paired unseen deltaRMSE difference (dB); negative favors first arm')
            ax.set_ylim(-.7, len(comparisons) - .3); export(fig, folder / f'{split}_paired_scene_differences')
            atomic(folder / f'{split}_paired_scene_differences.source.json', dict(comparisons=comparisons,
                  intervals='95% percentile interval from 2000 paired scene resamples; one training seed'))
    atomic(folder / 'figure_contract.json', dict(backend='python', training_curve_claim='Observed optimization trajectory only; T2/T3 have different velocity targets and state distributions',
        fixed_scene_claim='Qualitative conditional response of a hash-preselected held-out scene',
        gates_claim='Input support and nonlocal output diagnostic; not proof of reflection paths',
        uncertainty='One training seed; curves have no across-seed intervals; final bootstrap is scene-paired',
        smoke_debug_only=args.smoke, explicit_single_arm=args.arm,
        exclusions='Only hash-fixed scenes in example images; complete quantitative dataset retained',
        source_data='JSON logs and exact saved NPZ prediction arrays', automatic_QA='Alignment, PDF text and PDF collisions per render'))


if __name__ == '__main__': main()
