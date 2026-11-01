"""Render an audited source-only DWT study and execute its reader notebook."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pywt
from nbclient import NotebookClient
from scipy.stats import norm

from visualize_source_suite import COLORS, SOURCES, frame, save, write

plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 10})


def rows(root, name):
    with (root/name).open() as f:
        return list(csv.DictReader(f))


def main(root):
    info = json.loads((root/'analysis.json').read_text())
    energy = rows(root, 'energy_summary.csv')
    variances = rows(root, 'analytic_haar_variance.csv')
    covs = np.load(root/'analytic_covariance.npz')
    probes = np.load(root/'fixed_coefficient_draws.npz')
    samples = np.load(root/'samples.npy', mmap_mode='r')
    print('Loaded analysis', flush=True)
    fig, axes = frame('Matérn source: Haar scale allocation',
        '256 × 256 fields; 512 independent draws per source; no centering or per-field normalization', cols=2)
    for name, color in zip(SOURCES, COLORS):
        rr = [r for r in energy if r['source'] == name]
        rr = [next(r for r in rr if r['band'] == b) for b in info['band_order']]
        axes[0].plot(range(7), np.array(info['theory_energy'][name])*100, color=color, label=name)
        axes[0].errorbar(range(7), [float(r['pooled_mc_fraction'])*100 for r in rr],
            yerr=[float(r['mc_standard_error'])*200 for r in rr], fmt='o', ms=3, capsize=2, color=color)
        v = [next(float(r['variance']) for r in variances
             if r['source'] == name and int(r['level']) == j and r['orientation'] == 'V') for j in range(1,7)]
        axes[1].semilogy(range(1,7), v, '-o', color=color, ms=4)
    axes[0].set(title='a  Total band energy', ylabel='Fraction of expected field energy (%)',
                xlabel='Detail bands (finest → coarsest), then approximation', xticks=range(7), xticklabels=info['band_order'])
    axes[1].set(title='b  Variance of one coefficient', xlabel='Haar detail level j (support: 2ʲ pixels)',
                ylabel='Theoretical variance, column-difference coefficient', xticks=range(1,7))
    for ax in axes: ax.grid(axis='y', alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5)
    save(fig, axes, root/'energy_allocation', dict(draws=512, uncertainty='dots: pooled Monte Carlo ratio ±2 MC SE; lines: theory',
        energy_definition='sum of squared dimensionless source amplitudes; not electromagnetic energy'))

    c = covs['MS']; sd = np.sqrt(np.diag(c)); corr = c/np.outer(sd,sd)
    fig, axes = frame('Gaussian coefficients, non-independent scales',
        'Fixed spatial probes across independent MS draws; Cj = column-difference at level j', cols=2, height=5.3)
    q = norm.ppf((np.arange(512)+.5)/512)
    for j, color in zip((0,2,4), (COLORS[1],COLORS[2],COLORS[4])):
        axes[0].plot(q, np.sort(probes['MS'][:,j]/sd[j]), '.', ms=3, color=color, label=f'C{j+1}')
    axes[0].plot([-3.4,3.4],[-3.4,3.4], color='#777777', lw=1, ls='--')
    axes[0].set(title='a  Standardized Gaussian Q–Q', xlabel='Standard-normal theoretical quantile',
        ylabel='Coefficient / theoretical standard deviation')
    axes[0].legend(loc='upper left')
    im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    labels = [f'C{j}' for j in range(1,7)]+['C3 nbr']
    axes[1].set(title='b  Exact MS correlation (selected probes)', xticks=range(7), yticks=range(7),
                xticklabels=labels, yticklabels=labels)
    for i in range(7):
        for j in range(7):
            axes[1].text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',fontsize=8,
                color='white' if abs(corr[i,j])>.65 else '#222222')
    cb = fig.colorbar(im, cax=fig.add_axes([.70,.09,.23,.022]), orientation='horizontal')
    cb.set_label('Correlation (theoretical, not fitted)', fontsize=9)
    save(fig, axes, root/'gaussian_dependence', dict(probe_origin=[128,128], neighbor_origin=[128,136],
        dependence_scope='selected aligned probes, not a complete wavelet covariance', correlation=corr.tolist()))

    x = samples[4,0].astype(np.float64)
    coeff = pywt.wavedec2(x, 'haar', mode='periodization', level=6)
    def reconstruct(levels, approximation=False):
        selected = [coeff[0] if approximation else np.zeros_like(coeff[0])]
        selected += [tuple(a if 7-i in levels else np.zeros_like(a) for a in triplet)
                     for i,triplet in enumerate(coeff[1:],1)]
        return pywt.waverec2(selected,'haar',mode='periodization')
    coarse, middle, fine = reconstruct([],True), reconstruct([4,5,6]), reconstruct([1,2,3])
    reconstruction_error = float(np.max(np.abs(x-coarse-middle-fine)))
    assert reconstruction_error < 1e-12
    np.savez_compressed(root/'reconstructed_components.npz', full=x, A6=coarse, D4_D6=middle, D1_D3=fine)
    fig, axes = plt.subplots(1,4,figsize=(14,4.4))
    fig.subplots_adjust(left=.025,right=.975,top=.76,bottom=.20,wspace=.10)
    fig.suptitle('One MS source realization, separated into Haar components',y=.98,fontsize=14,fontweight='bold')
    fig.text(.5,.89,'Full field = A6 + D4–D6 + D1–D3; common color scale; no target map or conditioning',ha='center')
    for ax,field,title in zip(axes,(x,coarse,middle,fine),('a  Full MS source','b  Approximation A6','c  Details D4–D6','d  Details D1–D3')):
        im=ax.imshow(field,cmap='RdBu_r',vmin=-3,vmax=3,interpolation='nearest');ax.set_title(title);ax.set_axis_off()
    cb=fig.colorbar(im,cax=fig.add_axes([.30,.11,.40,.025]),orientation='horizontal',extend='both')
    cb.set_label('Dimensionless source amplitude (shared scale; ±3 display clipping)',fontsize=9)
    save(fig,axes,root/'multiscale_reconstruction',dict(draw_index=0,reconstruction_error=reconstruction_error,
        color_scale=[-3,3], clipping='display only', coefficients='Haar, no thresholding'))

    # The loss averages 2×2 BEFORE its two orthonormal Haar levels.
    down = x.reshape(128,2,128,2).mean((1,3))
    native = pywt.wavedec2(x,'haar',level=3)
    after = pywt.wavedec2(down,'haar',level=2)
    loss_error = max(float(np.max(np.abs(a-b/2)))
        for at,bt in zip(after[1:],native[1:3]) for a,b in zip(at,bt))
    assert loss_error < 1e-12
    write(root/'additional_checks.json',dict(component_reconstruction_error=reconstruction_error,
        loss_mapping_error=loss_error, loss_mapping='loss haar1 = native D2/2; loss haar2 = native D3/2; valid complete supports only'))

    table='\n'.join(f'| {b} | {100*info["theory_energy"]["W"][i]:.3f}% | {100*info["theory_energy"]["MS"][i]:.3f}% |'
                    for i,b in enumerate(info['band_order']))
    cond = [r for r in rows(root,'conditional_gaussian.csv') if r['source']=='MS']
    report = f'''# DWT analysis of the current Matérn source distribution

## Conclusions and scope

MS is already a multiscale Gaussian field, not noise with a single correlation scale. Its Haar coefficients remain jointly Gaussian, but their variances vary by scale and the coefficients are generally dependent. MS provides multiscale correlation, not building-aware reflection or shadow locations.

This analysis calls the actual `model/matern_source.py` implementation and uses 512 independent 256x256 samples for each of W/S2/S8/S32/MS. Groups share paired base noise. The source-spectrum SHA256 matches the current no-EMA MS training record. No model, source definition, training, or test result was changed. The primary analysis uses six Haar levels; three db4 levels assess basis and boundary sensitivity only. Units are pixels and dimensionless squared amplitudes, not meters or electromagnetic energy. Inputs are not centered or normalized per image.

## Source distribution and wavelet covariance

The code uses nu=1 and lambda in {{2,8,32}} pixels, with C_lambda(r)=(r/lambda)K_1(r/lambda) and C_lambda(0)=1. This is not the length-scale convention containing sqrt(2nu).

The covariance is C_MS=(C_2+C_8+C_32)/3. The spectra are averaged before taking the square root for sampling; the method neither selects one kernel at random nor averages three spectral amplitudes. A positive-definite 1024-side circulant embedding is cropped to 256, avoiding direct wraparound adjacency at the output image edges.

Let Q be an orthogonal Haar transform. Then w=Qx_0 is distributed as N(0, Q C_MS Q^T). Each coefficient variance is psi^T C_MS psi, a weighted sum of the covariance spectrum and squared wavelet-filter response. For white noise C=I, QIQ^T=I; for colored sources, this matrix is generally not diagonal. An orthogonal basis does not imply independent random coefficients.

Every coefficient at a fixed scale, orientation, and position is zero-mean Gaussian, and its square follows a variance-scaled chi-squared distribution with one degree of freedom. Many small values do not establish true sparsity, and no mechanism automatically aligns with building edges. Isotropy gives equal H/V variances at corresponding positions; the HH filter response differs and need not have the same variance.

## Energy across scales

The table gives the analytic expected energy ratio E[band energy]/E[whole-image energy], checked against the pooled ratio from 512 Monte Carlo draws. Each D_j includes three detail orientations; A6 counts only the final low-frequency band to avoid counting intermediate low-frequency bands repeatedly.

| Haar band | W | MS |
|---|---:|---:|
{table}

D1-D6 have original-image filter support widths of 2, 4, 8, 16, 32, and 64 pixels. These are neither strict wavelength bands nor the Matérn lambda values. Each of the three kernels contributes one-third of the covariance, not one-third of the energy in a DWT scale. The two finest MS levels contain 15.39% of energy versus 93.75% for W; MS has substantially more medium- and coarse-scale energy, but no hard high-frequency cutoff.

![Energy and coefficient variance](energy_allocation.png)

Across 512 MS samples, the pooled final low-frequency A6 fraction is 24.98%, compared with 24.83% analytically. Distinguish the pooled ratio from the mean of per-image ratios. Figure error bars show plus or minus two standard errors of the Monte Carlo estimator, not the standard deviation across individual images. The maximum Parseval relative error across groups is about 2x10^-15.

## Cross-scale dependence and apparent heavy tails

Covariances are calculated exactly and checked against samples for vertical-detail probes C1-C6 aligned at (128,128), plus one adjacent C3 probe. These represent only the seven specified probes, not all positions or full conditional prediction ability.

For MS, conditioning on only C4/C5/C6 explains {100*float(cond[0]['analytic_explained_variance']):.3f}%, {100*float(cond[1]['analytic_explained_variance']):.3f}%, and {100*float(cond[2]['analytic_explained_variance']):.3f}% of the theoretical variance in C1/C2/C3, respectively. Cross-scale dependence exists, but these few coarse coefficients cannot determine the details. The Gaussian conditional mean is Sigma_fc Sigma_cc^-1 w_c, and the conditional covariance is Sigma_ff-Sigma_fc Sigma_cc^-1 Sigma_cf. This does not automatically define a strict parent-child Markov tree.

Another observation: directly pooling raw coefficients from six scales at fixed positions gives MS sample excess kurtosis near 8.06; scaling first by each scale's theoretical standard deviation gives about 0.08. The peaked, heavy-tailed raw histogram is largely explained by a mixture of Gaussians with different variances, not evidence that the source itself is non-Gaussian. Independent field draws are the statistical units; adjacent coefficients within one image are not assumed independent.

![Gaussianity and dependence](gaussian_dependence.png)

## Basis, boundary, and current-loss correspondence

`db4_sensitivity.csv` reports internal-coefficient second moments and sampling standard errors at three db4 levels under periodization and symmetric extension. Seven coefficients are trimmed on each side to limit boundary contamination; downsampling alignment can also differ between modes, so not every difference is a boundary effect. Periodization conserves full-coefficient energy. Symmetric extension has redundant boundary coefficients, so its full sum of squared coefficients is not directly a partition into exclusive energy fractions. Exact values depend on the basis; Haar percentages are not intrinsic, exact frequency-band percentages of the data. The conclusion that linear transforms preserve Gaussianity does not depend on Haar or db4.

The current `training/detail_losses.py:14` first applies 2x2 averaging, then computes two Haar levels. On fully valid support, loss haar1 corresponds to original-image D2/2 and haar2 to D3/2, up to orientation naming and ordering; the finest original-image D1 is excluded. The maximum numerical verification error is {loss_error:.3g}. The actual loss also filters by valid support and averages over coefficients, so it cannot be directly compared by division with this report's unmasked energy fractions. The wavelet loss concerns velocity error, whereas this analysis concerns the source; they are not interchangeable.

## Implications for CKM design

MS changes the multiscale covariance of initial errors and random perturbations: far fewer independent finest-scale fluctuations than W, and more medium- and coarse-scale variance. The fixed kernel remains stationary, isotropic, and environment-independent. DWT cannot by itself align random ripples with buildings, Tx, or enhancement bands.

Thus, a multiscale source and the need for condition-driven multiscale velocity processing are compatible. A possible direction is to predict signed fine-scale corrections or conditional means from coarse propagation states and environmental features, while retaining controlled residual uncertainty. Simply increasing detail variance may add incorrect texture. Source covariance alone cannot demonstrate better recovery of electromagnetic reflections; corresponding signed target/prediction wavelet errors, local spatial coherence, and propagation-region evaluations are also needed. No additional training was conducted, and no such improvement is claimed here.

![Invertible decomposition of one source field](multiscale_reconstruction.png)

## Reproduction and files

- `sampling.json`: source specification, hashes, and sampling namespace; `samples.npy`: complete source draws.
- `analytic_haar_variance.csv` and `energy_summary.csv`: analytic and Monte Carlo energies.
- `fixed_coefficient_statistics.csv` and `conditional_gaussian.csv`: fixed-probe statistics and conditional variance.
- `matern_dwt_analysis.ipynb`: executed notebook.
- Generation scripts: `scripts/sample_matern_dwt.py`, `scripts/analyze_matern_dwt.py`, and `scripts/render_matern_dwt.py`.
- Figures are exported as PNG/PDF/SVG with alignment, font, and collision audits.

References: [PyWavelets 2D DWT definition](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html), [boundary extension modes](https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html), and [GPML covariance functions and Matérn kernels](https://gaussianprocess.org/gpml/chapters/RW4.pdf).
'''
    (root/'README.md').write_text(report)
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = dict(display_name='CKM Python 3.12',language='python',name='ckm-python312')
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb.cells = [md('# Matérn source distribution: DWT multiscale energy and coefficient dependence\n\nActual SourceSampler with five groups of 512 independent draws each. The analysis does not use target labels or start training.'),
        code("from pathlib import Path\nimport json, csv\nimport numpy as np\nimport pywt\nfrom IPython.display import display, Image\nroot = Path.cwd()\ninfo = json.loads((root/'analysis.json').read_text())\nsamples = np.load(root/'samples.npy', mmap_mode='r')\nprint(samples.shape, info['units'])"),
        md('## 1. Current source definition\nnu=1 and lambda=2/8/32 pixels; sample from the square root of the equally mixed covariances. $w=Qx_0\\sim\\mathcal N(0,QCQ^T)$. An orthogonal basis does not make colored-field coefficients independent.'),
        code("receipt = json.loads((root/'sampling.json').read_text())\nprint(json.dumps(receipt, ensure_ascii=False, indent=2)[:12000])"),
        md('## 2. Invertible transform and energy conservation\nD1 is the finest level; count only final A6 to avoid duplicate low-frequency energy.'),
        code("x = samples[4,0].astype(float)\nc = pywt.wavedec2(x, 'haar', mode='periodization', level=6)\nr = pywt.waverec2(c, 'haar', mode='periodization')\nenergy = np.sum(c[0]**2) + sum(np.sum(a*a) for triplet in c[1:] for a in triplet)\nprint('reconstruction max error:', np.max(np.abs(x-r)))\nprint('Parseval relative error:', abs(energy/np.sum(x*x)-1))\nprint('MS theoretical band fractions:', dict(zip(info['band_order'], info['theory_energy']['MS'])))\ndisplay(Image(filename=str(root/'energy_allocation.png'), width=1100))"),
        md('## 3. Gaussianity and dependence across independent draws at fixed positions\nAdjacent coefficients in one image are not independent samples. Mixing scale-dependent variances can produce a heavy-tailed histogram.'),
        code("print(json.dumps(info['checks']['MS'], indent=2))\nwith (root/'conditional_gaussian.csv').open() as f:\n    for row in csv.DictReader(f):\n        if row['source']=='MS': print(row)\ndisplay(Image(filename=str(root/'gaussian_dependence.png'), width=1100))"),
        md('## 4. Index correspondence with the current Haar loss\nAverage-downsample by two first; the two loss levels correspond to original-image D2/2 and D3/2, not D1/D2. Verify on fully valid support.'),
        code("down = x.reshape(128,2,128,2).mean((1,3))\na = pywt.wavedec2(down, 'haar', level=2)\nb = pywt.wavedec2(x, 'haar', level=3)\nprint(max(np.max(np.abs(u-v/2)) for at,bt in zip(a[1:],b[1:3]) for u,v in zip(at,bt)))\ndisplay(Image(filename=str(root/'multiscale_reconstruction.png'), width=1100))"),
        md('## 5. Boundary sensitivity and interpretation limits\nInternal db4 statistics appear below. Symmetric extension has redundant boundary coefficients, so its squared-coefficient sum is not an orthogonal energy partition. A stationary source provides multiscale correlation, not condition-aligned electromagnetic propagation structure. See README.md for the full discussion and references.'),
        code("with (root/'db4_sensitivity.csv').open() as f:\n    for row in csv.DictReader(f):\n        if row['source']=='MS': print(row)")]
    NotebookClient(nb,timeout=120,kernel_name='ckm-python312',resources={'metadata':{'path':str(root)}}).execute()
    nbformat.write(nb,root/'matern_dwt_analysis.ipynb')
    print('Figures, report, and executed notebook saved', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
