"""External FM sources only: zero mean, unit marginal variance, paired RNG."""
from functools import lru_cache
import hashlib
import time
import numpy as np
from scipy.special import k1
import torch

SOURCES = ('W', 'S2', 'S8', 'S32', 'MS')
LENGTHS = (2., 8., 32.)


def private_seed(metadata, namespace, seed=42):
    value = f"source-v1|{seed}|{namespace}|{metadata['scene_id']}|{float(metadata['steering_deg']):.6f}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big')


@lru_cache(maxsize=4)
def source_spectra(resolution=256):
    extent = 2 ** int(np.ceil(np.log2(2 * resolution)))
    while extent <= 2048:
        axis = np.minimum(np.arange(extent), extent - np.arange(extent))
        radius = np.hypot(axis[:, None], axis[None, :])
        spectra = []
        for length in LENGTHS:
            r = radius / length
            covariance = np.ones_like(r)
            covariance[r > 0] = r[r > 0] * k1(r[r > 0])
            spectra.append(np.fft.rfft2(covariance).real)
        values = np.array(spectra)
        if values.min() >= -1e-10:
            values = np.maximum(values, 0)
            # Mixture of covariance eigenvalues, NOT mixture of their square roots.
            all_values = np.concatenate((np.ones_like(values[:1]), values, values.mean(0, keepdims=True)))
            return extent, all_values
        extent *= 2
    raise RuntimeError('Non-PSD source embedding; no silent covariance approximation')


class SourceSampler:
    def __init__(self, source, resolution=256, seed=42):
        if source not in SOURCES:
            raise ValueError(source)
        self.source, self.resolution, self.seed = source, resolution, seed
        start = time.perf_counter()
        self.extent, eigenvalues = source_spectra(resolution)
        self.eigenvalues = eigenvalues[SOURCES.index(source)]
        self.amplitude = np.sqrt(self.eigenvalues).astype(np.float32)
        self.cold_build_seconds = time.perf_counter() - start
        self._devices = {}

    def specification(self):
        lengths = [] if self.source == 'W' else list(LENGTHS) if self.source == 'MS' else [float(self.source[1:])]
        return dict(source=self.source, resolution=self.resolution, seed=self.seed, external_source=True,
                    internal_noise=False, mean=0., marginal_variance=1., nu=None if self.source == 'W' else 1.,
                    length_pixels=lengths, length_convention='r/lambda_no_sqrt_2nu',
                    weights=[] if not lengths else [1/len(lengths)]*len(lengths),
                    embedding_size=self.extent, min_eigenvalue=float(self.eigenvalues.min()),
                    max_eigenvalue=float(self.eigenvalues.max()), zero_eigenvalues=int((self.eigenvalues==0).sum()),
                    amplitude_sha256=hashlib.sha256(self.amplitude.tobytes()).hexdigest(),
                    sampler='PSD_checked_circulant_embedding_common_real_white_crop',
                    per_sample_normalization=False)

    @torch.no_grad()
    def base_noise(self, metadata, device, namespace):
        device = torch.device(device)
        return torch.stack([torch.randn((1,self.extent,self.extent),device=device,dtype=torch.float32,
            generator=torch.Generator(device=device).manual_seed(private_seed(m,namespace,self.seed))) for m in metadata])

    @torch.no_grad()
    def transform(self, white):
        if white.ndim != 4 or tuple(white.shape[1:]) != (1,self.extent,self.extent):
            raise ValueError('Wrong source base-noise shape')
        if self.source == 'W':
            field = white.float()
        else:
            key = str(white.device)
            if key not in self._devices:
                self._devices[key] = torch.from_numpy(self.amplitude).to(white.device)
            with torch.autocast(device_type=white.device.type,enabled=False):
                field = torch.fft.irfft2(torch.fft.rfft2(white.float(),norm='ortho')*self._devices[key],
                                        s=(self.extent,self.extent),norm='ortho')
        return field[..., :self.resolution, :self.resolution].contiguous()

    def sample(self, metadata, *, device, namespace):
        return self.transform(self.base_noise(metadata,device,namespace))


def path_triplet(x0, target, t):
    if x0.shape != target.shape or t.shape != (x0.shape[0],):
        raise ValueError('Invalid endpoint/time shapes')
    time = t[:,None,None,None]
    return (1-time)*x0 + time*target, target-x0
