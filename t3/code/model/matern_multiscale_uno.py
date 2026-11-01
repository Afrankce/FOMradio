"""Trajectory-frozen Matérn innovations inside the existing U-NO encoder.

The external FM source remains iid N(0,I). These are auxiliary latent fields,
not a redefinition of p0. Lengths use C(r)=(r/lambda) K_1(r/lambda), nu=1,
the GaussianRandomFields.jl convention (no sqrt(2 nu) inside the argument).
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
from scipy.special import k1
import torch
from torch import nn
from torch.nn import functional as F

from model.attention_multiscale_uno import AttentionMultiscaleUNO2d

VARIANTS = ("B2", "M1", "M2", "M3")


def latent_seed(metadata, namespace):
    key = f"matern-v1|42|{namespace}|{metadata['scene_id']}|{float(metadata['steering_deg']):.6f}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


@lru_cache(maxsize=32)
def embedding_spectra(size, level):
    """Common embedding for paired white/M1/M2 innovations; reject non-PSD CE.

Both length schedules are checked, so every variant consumes identical base
white tensors. Cropping the extended field avoids periodic wrap on the ROI.
Only floating-point negative eigenvalues (absolute <=1e-10) may be clipped.
"""
    schedules = np.array([[2., 4., 8.]]) * np.array([[2. ** -level], [1.]])
    extent = 2 ** int(np.ceil(np.log2(max(2 * size, 32))))
    while extent <= 2048:
        axis = np.minimum(np.arange(extent), extent - np.arange(extent))
        distance = np.hypot(axis[:, None], axis[None, :])
        spectra = []
        for lengths in schedules:
            group = []
            for length in lengths:
                radius = distance / length
                covariance = np.ones_like(radius)
                positive = radius > 0
                covariance[positive] = radius[positive] * k1(radius[positive])
                group.append(np.fft.rfft2(covariance).real)
            spectra.append(group)
        values = np.array(spectra)
        if values.min() >= -1e-10:
            return extent, np.sqrt(np.maximum(values, 0)).astype(np.float32), float(values.min())
        extent *= 2
    raise RuntimeError("Matérn circulant embedding is not PSD within the registered padding budget")


class MaternBankSampler:
    def __init__(self, variant, resolution=256):
        if variant not in VARIANTS or resolution % 16:
            raise ValueError("Invalid variant or resolution")
        self.variant = variant
        self.resolution = resolution
        self._device_spectra = {}

    def specification(self):
        return [dict(level=s, size=self.resolution // 2 ** s,
                     embedding_size=embedding_spectra(self.resolution // 2 ** s, s)[0],
                     min_eigenvalue=embedding_spectra(self.resolution // 2 ** s, s)[2],
                     lambda_domain_fraction=None if self.variant == "B2" else [
                         v / self.resolution * (1 if self.variant == "M1" else 2 ** s)
                         for v in (2, 4, 8)]) for s in range(5)]

    @torch.no_grad()
    def sample(self, metadata, *, device, namespace):
        # A private, per-scene generator leaves x0, t, CFG and loader RNG untouched.
        device = torch.device(device)
        generators = [torch.Generator(device=device).manual_seed(latent_seed(m, namespace)) for m in metadata]
        result = []
        for level in range(5):
            size = self.resolution // 2 ** level
            extent, spectra, _ = embedding_spectra(size, level)
            white = torch.stack([torch.randn((3, extent, extent), device=device, generator=g)
                                 for g in generators])
            if self.variant == "B2":
                field = white
            else:
                key = (level, str(device))
                if key not in self._device_spectra:
                    self._device_spectra[key] = torch.from_numpy(spectra[0 if self.variant == "M1" else 1]).to(device)
                # Orthonormal forward/inverse and sqrt(eigenvalue), not the PSD.
                field = torch.fft.irfft2(torch.fft.rfft2(white, norm="ortho") * self._device_spectra[key],
                                        s=(extent, extent), norm="ortho")
            result.append(field[..., :size, :size].contiguous())
        return tuple(result)


class InnovationInjection(nn.Module):
    def __init__(self, channels, condition_channels, adaptive):
        super().__init__()
        self.adaptive = adaptive
        self.head = nn.Sequential(nn.Conv2d(channels + condition_channels, 16, 1), nn.GELU(), nn.Conv2d(16, 3, 1))
        self.time_head = nn.Linear(512, 3)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.time_head.weight)
        nn.init.zeros_(self.time_head.bias)
        self.project = nn.Conv2d(1, channels, 1, bias=False)
        self.gain = nn.Parameter(torch.tensor(0.05))
        if not adaptive:
            self.head.requires_grad_(False)
            self.time_head.requires_grad_(False)

    def weights(self, value, condition, time):
        if not self.adaptive:
            return torch.full((value.shape[0], 3, *value.shape[-2:]), 1 / 3, device=value.device, dtype=torch.float32)
        logits = self.head(torch.cat((value, condition), dim=1)) + self.time_head(time)[:, :, None, None]
        return logits.float().softmax(dim=1)

    def forward(self, value, condition, time, bank):
        if bank.shape != (value.shape[0], 3, *value.shape[-2:]):
            raise ValueError("Innovation bank does not match encoder stage")
        alpha = self.weights(value, condition, time)
        innovation = (alpha.sqrt() * bank).sum(dim=1, keepdim=True).to(value.dtype)
        return value + self.gain * self.project(innovation)


class MaternMultiscaleUNO2d(AttentionMultiscaleUNO2d):
    def __init__(self, variant, **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(**kwargs)
        self.variant = variant
        self.injections = nn.ModuleList([InnovationInjection(c, e, variant == "M3")
                                       for c, e in zip(self.state_channels, self.encoder_features[:5])])
        self._innovation_bank = None

    @contextmanager
    def use_innovations(self, bank):
        if self._innovation_bank is not None or len(bank) != 5:
            raise RuntimeError("Nested or invalid innovation bank")
        self._innovation_bank = bank
        try:
            yield
        finally:
            self._innovation_bank = None

    def _velocity(self, condition, state, step, embeddings=None):
        if self._innovation_bank is None:
            raise RuntimeError("Sample and freeze auxiliary fields before evaluating velocity")
        self._validate_state_and_condition(condition, state)
        conditions = self._validate_embeddings(self.embed_model(condition) if embeddings is None else embeddings, state)
        grid_x, grid_y = self.coordinate_grid(state)
        value = self.lifting(torch.cat((state, condition, grid_x, grid_y), dim=1))
        time = self._time_embedding(self._normalize_step(step, state))
        skips = []
        for s, (stage, downsample) in enumerate(zip(self.encoder_stages, self.downsamples)):
            # s>0: value is the downsampled output of the PREVIOUS FNO stage.
            value = self.injections[s](value, conditions[s], time, self._innovation_bank[s])
            value = stage(value, conditions[s], time)
            skips.append(value)
            value = downsample(value)
        value = self.injections[4](value, conditions[4], time, self._innovation_bank[4])
        value = self.bottleneck(value, conditions[4], time)
        for index, level in enumerate(reversed(range(4))):
            value = self.upsample_fusions[index](value, skips[level])
            value = self.decoder_stages[index](value, conditions[level], time)
        return self.projection_output(F.gelu(self.projection_hidden(value)))
