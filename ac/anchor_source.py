"""Finite Beam-guided noise-anchor sampling for the exploratory A ablation."""
import math

import torch
from torch.nn import functional as F

from beam_matern_source import filter_noise, probability_map
from beam_scale_source import BeamScaleMaternSource, combine_scales
from model.matern_source import private_seed


def sample_anchor_counts(q, metadata, count, seed, namespace):
    """Draw reproducible per-sample discrete anchor counts on CPU.

    q is deliberately detached: the initial study uses a fixed q rule, while
    the scale-weight network is differentiable.  The fixed CPU RNG also avoids
    GPU scatter nondeterminism and preserves independent streams per query.
    """
    if q.ndim != 4 or q.shape[1] != 1 or len(metadata) != q.shape[0]:
        raise ValueError("q and metadata must describe a batch of single-channel maps")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("anchor count must be a positive integer")
    probs = q.detach().float().cpu().flatten(1)
    if not bool(torch.isfinite(probs).all()) or bool((probs < 0).any()):
        raise ValueError("q must contain finite, nonnegative probabilities")
    if not torch.allclose(probs.sum(1), torch.ones(probs.shape[0]), atol=1e-5, rtol=1e-5):
        raise ValueError("q must sum to one for each sample")
    n = probs.shape[1]
    rows = []
    for p, m in zip(probs, metadata):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(private_seed(m, namespace + "-anchors", seed))
        indices = torch.multinomial(p, count, replacement=True, generator=generator)
        rows.append(torch.bincount(indices, minlength=n))
    return torch.stack(rows).reshape(q.shape).to(device=q.device)


class AnchorMaternSource(BeamScaleMaternSource):
    """Finite anchors, fixed q rule and fixed or learned scale weights."""

    def __init__(self, seed, gradient_scale, resolution=256, anchor_count=4096,
                 mix_strength=0.5, q_mode="uniform", learn_scales=False,
                 scale_floor=0.05):
        if q_mode not in ("uniform", "beam"):
            raise ValueError("q_mode must be uniform or beam")
        if not isinstance(anchor_count, int) or anchor_count <= 0:
            raise ValueError("anchor_count must be positive")
        if not math.isfinite(float(mix_strength)) or not 0.0 <= mix_strength <= 1.0:
            raise ValueError("mix_strength must be finite and in [0, 1]")
        super().__init__(seed=seed, gradient_scale=gradient_scale, resolution=resolution,
                         scale_floor=scale_floor, learn_spatial_weights=False)
        self.seed = int(seed)
        self.anchor_count = anchor_count
        self.mix_strength = float(mix_strength)
        self.q_mode = q_mode
        self.learn_scales = bool(learn_scales)
        for p in self.scale_net.parameters():
            p.requires_grad_(self.learn_scales)

    def forward(self, beam, metadata, namespace):
        self._validate_beam(beam)
        if len(metadata) != beam.shape[0]:
            raise ValueError("metadata must contain one entry per beam")
        with torch.autocast(device_type=beam.device.type, enabled=False):
            beam = beam.float()
            weights = self.scale_weights(beam)
            alpha, beta = F.softplus(self.spatial.a), F.softplus(self.spatial.d)
            q = probability_map(beam, alpha, beta, self.spatial.gradient_scale)
            r, extent = self.spatial.resolution, self.spatial.extent
            if self.q_mode == "uniform":
                q = torch.full_like(q, 1.0 / (r * r))
            counts = sample_anchor_counts(q, metadata, self.anchor_count, self.seed, namespace)
            rho = ((1.0 - self.mix_strength) + self.mix_strength * (r * r)
                   * counts.float() / self.anchor_count)
            rho = F.pad(rho - 1.0, (0, extent - r, 0, extent - r)) + 1.0
            rho_fft = torch.fft.rfft2(rho, norm="backward")
            fields, variances = [], []
            for i, key in enumerate(("S2", "S8", "S32")):
                eps = self.spatial.samplers[key].base_noise(
                    metadata, beam.device, namespace + "-" + key
                )
                filtered = filter_noise(rho.sqrt() * eps, self.spatial.amplitudes[i])
                variance = torch.fft.irfft2(
                    rho_fft * self.scale_var_kernels[i],
                    s=(extent, extent), norm="backward",
                ).clamp_min(0.0)
                fields.append(filtered[..., :r, :r])
                variances.append(variance[..., :r, :r])
            guided, _ = combine_scales(
                torch.cat(fields, dim=1), torch.cat(variances, dim=1), weights
            )
            base = self.spatial.samplers["MS"].sample(
                metadata, device=beam.device, namespace=namespace + "-base"
            )
            return math.sqrt(0.75) * base + 0.5 * guided, q, weights, counts
