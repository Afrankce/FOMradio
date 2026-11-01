"""Exact Haar covariance plus Monte Carlo DWT analysis of the actual FM prior."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pywt
from scipy.stats import skew,kurtosis

SOURCES=('W','S2','S8','S32','MS')
J=6
FEATURES=[f'C{j}' for j in range(1,J+1)]+['C3_neighbor']


def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))


def csv_out(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader();w.writerows(rows)


def haar_atom(extent,j,orientation,neighbor=False):
    n=2**j;lo=np.ones(n)/np.sqrt(n);hi=lo.copy();hi[n//2:]*=-1
    a,b={'A':(lo,lo),'H':(hi,lo),'V':(lo,hi),'D':(hi,hi)}[orientation]
    field=np.zeros((extent,extent));start=128;column=128+(n if neighbor else 0)
    field[start:start+n,column:column+n]=a[:,None]*b[None,:]
    return field


def analyze(out):
    samples=np.load(out/'samples.npy',mmap_mode='r');spectra=np.load(out/'embedding_spectra.npy')
    info=json.loads((out/'sampling.json').read_text());assert info['status']=='complete'
    draws=info['draws'];N=256;extent=info['specifications']['MS']['embedding_size']
    assert samples.shape==(5,draws,N,N)
    atoms={f'{o}{j}':haar_atom(extent,j,o) for j in range(1,J+1) for o in ('H','V','D')}
    atoms['A6']=haar_atom(extent,6,'A');atoms['V3_neighbor']=haar_atom(extent,3,'V',True)
    transforms={k:np.fft.rfft2(x,norm='ortho') for k,x in atoms.items()};del atoms
    multiplicity=np.ones((extent,extent//2+1));multiplicity[:,1:-1]=2
    variance={};covariances={};theory=[];energy_theory={}
    selected=np.stack([transforms[f'V{j}'].ravel() for j in range(1,J+1)]+[transforms['V3_neighbor'].ravel()])
    for i,s in enumerate(SOURCES):
        variance[s]={key:float(np.sum(spectra[i]*multiplicity*abs(f)**2)) for key,f in transforms.items()}
        weighted=selected*np.sqrt((spectra[i]*multiplicity).ravel())[None,:]
        covariance=(weighted.conj()@weighted.T).real
        covariances[s]=covariance
        expected=[]
        for j in range(1,J+1):
            expected.append(sum(variance[s][f'{o}{j}'] for o in ('H','V','D'))/(4**j))
            for o in ('H','V','D'):
                theory.append(dict(source=s,level=j,orientation=o,variance=variance[s][f'{o}{j}'],
                    coefficient_count=(N//2**j)**2,expected_energy_fraction=variance[s][f'{o}{j}']/(4**j)))
        expected.append(variance[s]['A6']/(4**6));energy_theory[s]=expected
        assert abs(sum(expected)-1)<1e-9,(s,expected)
        if s=='W':assert np.max(abs(covariance-np.eye(7)))<1e-12
    assert np.max(abs(covariances['MS']-sum(covariances[s] for s in ('S2','S8','S32'))/3))<1e-9
    csv_out(out/'analytic_haar_variance.csv',theory)
    np.savez_compressed(out/'analytic_covariance.npz',**covariances)
    del selected,transforms,multiplicity,spectra

    per_draw=[];summary=[];fixed_stats=[];sensitivity=[];feature_samples={};checks={};conditional=[]
    for i,s in enumerate(SOURCES):
        x=np.asarray(samples[i],dtype=np.float64)
        c=pywt.wavedec2(x,'haar',mode='periodization',level=J,axes=(-2,-1))
        power=np.sum(x*x,axis=(-2,-1));energies=[];probes=[]
        for j in range(1,J+1):
            details=c[-j];e=sum(np.sum(v*v,axis=(-2,-1)) for v in details);energies.append(e)
            pos=128//2**j;probes.append(details[1][:,pos,pos])
            for o,v in zip(('H','V','D'),details):
                probe=v[:,pos,pos];sd=np.sqrt(variance[s][f'{o}{j}']);z=probe/sd
                fixed_stats.append(dict(source=s,level=j,orientation=o,draws=draws,
                    analytic_variance=sd**2,empirical_variance=float(probe.var(ddof=1)),
                    standardized_mean=float(z.mean()),standardized_sd=float(z.std(ddof=1)),
                    skewness=float(skew(z,bias=False)),excess_kurtosis=float(kurtosis(z,bias=False)),
                    fraction_abs_z_gt3=float(np.mean(abs(z)>3)),coefficient_row=pos,coefficient_column=pos))
        energies.append(np.sum(c[0]**2,axis=(-2,-1)))
        energies=np.stack(energies,axis=1)
        pos=128//8;probes.append(c[-3][1][:,pos,pos+1]);probes=np.stack(probes,axis=1)
        feature_samples[s]=probes
        np.save(out/f'empirical_correlation_{s}.npy',np.corrcoef(probes,rowvar=False))
        parseval=float(np.max(abs(energies.sum(1)-power)/power))
        reconstructed=pywt.waverec2(c,'haar',mode='periodization',axes=(-2,-1))
        reconstruction=float(np.max(abs(reconstructed-x)))
        assert parseval<1e-12 and reconstruction<1e-12
        # Verify the pixel-space Haar atom agrees in sign and normalization with PyWavelets.
        for j in range(1,J+1):
            n=2**j;a=x[:,128:128+n,128:128+n]
            explicit=(a[:,:,:n//2].sum((1,2))-a[:,:,n//2:].sum((1,2)))/n
            assert np.allclose(explicit,probes[:,j-1],atol=1e-11,rtol=1e-11)
        labels=[f'D{j}' for j in range(1,J+1)]+['A6']
        for b,label in enumerate(labels):
            pooled=float(energies[:,b].sum()/power.sum())
            se=float(np.std(energies[:,b]-pooled*power,ddof=1)/np.sqrt(draws)/np.mean(power))
            summary.append(dict(source=s,band=label,draws=draws,
                analytic_fraction=energy_theory[s][b],pooled_mc_fraction=pooled,mc_standard_error=se,
                mean_per_draw_fraction=float(np.mean(energies[:,b]/power)),
                sd_per_draw_fraction=float(np.std(energies[:,b]/power,ddof=1))))
            per_draw.extend(dict(source=s,draw=d,band=label,energy=float(energies[d,b]),
                total_energy=float(power[d]),fraction=float(energies[d,b]/power[d])) for d in range(draws))
        covariance=covariances[s]
        for fine in (0,1,2):
            coarse=np.arange(3,6)
            beta=np.linalg.solve(covariance[np.ix_(coarse,coarse)],covariance[coarse,fine])
            r2=float(covariance[fine,coarse]@beta/covariance[fine,fine])
            conditional.append(dict(source=s,fine=f'C{fine+1}',condition='C4,C5,C6 at nested origin128',
                analytic_explained_variance=r2,conditional_variance=float(covariance[fine,fine]*(1-r2))))
        checks[s]=dict(parseval_max_relative_error=parseval,reconstruction_max_absolute_error=reconstruction,
            empirical_marginal_second_moment=float(power.mean()/N**2),sample_mean_mean=float(x.mean((-2,-1)).mean()),
            sample_mean_sd=float(x.mean((-2,-1)).std(ddof=1)),
            pooled_raw_cross_scale_excess_kurtosis=float(kurtosis(probes[:,:6].ravel(),bias=False)),
            normalized_cross_scale_excess_kurtosis=float(kurtosis((probes[:,:6]/np.sqrt(np.diag(covariance)[:6])).ravel(),bias=False)))
        # Longer wavelet sensitivity: boundary effects are not hidden in energy percentages.
        for mode in ('periodization','symmetric'):
            dc=pywt.wavedec2(x,'db4',mode=mode,level=3,axes=(-2,-1))
            ce=sum(np.sum(v*v,axis=(-2,-1)) for band in dc[1:] for v in band)+np.sum(dc[0]**2,axis=(-2,-1))
            for j in (1,2,3):
                for o,v in zip(('H','V','D'),dc[-j]):
                    interior=v[:,7:-7,7:-7]
                    vv=np.mean(interior**2,axis=(-2,-1))
                    sensitivity.append(dict(source=s,wavelet='db4',mode=mode,level=j,orientation=o,
                        full_coefficient_count=int(v.shape[-1]*v.shape[-2]),interior_count=int(interior.shape[-1]*interior.shape[-2]),
                        interior_second_moment=float(vv.mean()),mc_standard_error=float(vv.std(ddof=1)/np.sqrt(draws)),
                        coefficient_to_spatial_energy_ratio=float(np.sum(ce)/np.sum(power))))
            if mode=='periodization':assert np.max(abs(ce-power)/power)<1e-12
        print('DWT analyzed',s,flush=True)
    csv_out(out/'energy_per_draw.csv',per_draw);csv_out(out/'energy_summary.csv',summary)
    csv_out(out/'fixed_coefficient_statistics.csv',fixed_stats);csv_out(out/'db4_sensitivity.csv',sensitivity)
    csv_out(out/'conditional_gaussian.csv',conditional)
    np.savez_compressed(out/'fixed_coefficient_draws.npz',**feature_samples)
    write_json(out/'analysis.json',dict(status='complete',draws=draws,features=FEATURES,
        theory_energy=energy_theory,checks=checks,units='source pixels; dimensionless squared amplitudes, not RF energy',
        wavelet='Haar',levels=6,band_order=['D1','D2','D3','D4','D5','D6','A6'],
        primary_energy_estimator='ratio of summed band energy to summed full-field energy',
        field_centering=False,standardization='Only fixed-coefficient Gaussian diagnostics divide by theoretical SD',
        source_sha256=info['sampler_sha256'],analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    analyze(p.parse_args().out.resolve())
