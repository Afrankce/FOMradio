"""Isolated prototype: spatial Beam-conditioned weights on Matérn scales.

The existing BeamMaternSource supplies the probability map, filters, and
paired random-number streams.  This module does not alter the velocity model.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from beam_matern_source import (
    BeamMaternSource,
    filter_noise,
    probability_map,
    spatial_gradient,
)


def combine_scales(raw_fields, component_variance, weights, eps=1e-8):
    """Mix independent fields and normalize their analytic pointwise variance.

    Inputs have shape [batch, scales, height, width]. ``component_variance``
    describes the conditional variance of each unnormalized filtered field.
    The returned fractions refer to variance *after* weighting.
    """
    if raw_fields.shape != component_variance.shape or raw_fields.shape != weights.shape:
        raise ValueError("raw_fields, component_variance and weights must have the same shape")
    if raw_fields.ndim != 4:
        raise ValueError("expected [batch, scales, height, width]")
    weighted_variance = weights * component_variance
    total_variance = weighted_variance.sum(dim=1, keepdim=True).clamp_min(eps)
    field = (weights.sqrt() * raw_fields).sum(dim=1, keepdim=True)
    return field / total_variance.sqrt(), weighted_variance / total_variance


class BeamScaleMaternSource(nn.Module):
    """Add trainable, per-location weights to the existing three-scale source.

    Initial weights are exactly uniform. The Beam probability-map parameters
    are frozen by default so the first experiment isolates the scale weights.
    """

    def __init__(
        self,
        seed,
        gradient_scale,
        resolution=256,
        hidden_channels=16,
        scale_floor=0.05,
        learn_spatial_weights=False,
    ):
        super().__init__()
        if not math.isfinite(float(scale_floor)) or not 0.0 <= scale_floor < 1.0:
            raise ValueError("scale_floor must be finite and in [0, 1)")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.scale_floor = float(scale_floor)
        self.spatial = BeamMaternSource(seed, gradient_scale, resolution=resolution)
        self.spatial.a.requires_grad_(learn_spatial_weights)
        self.spatial.d.requires_grad_(learn_spatial_weights)
        self.scale_net = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)

        extent = self.spatial.extent
        kernels = torch.fft.irfft2(
            self.spatial.amplitudes, s=(extent, extent), norm="backward"
        )
        self.register_buffer(
            "scale_var_kernels", torch.fft.rfft2(kernels.square(), norm="backward")
        )

    def _validate_beam(self, beam):
        r = self.spatial.resolution
        if beam.ndim != 4 or beam.shape[1:] != (1, r, r):
            raise ValueError(f"beam must have shape [batch, 1, {r}, {r}]")

    def scale_weights(self, beam):
        """Return positive, per-pixel weights over the 2/8/32-pixel scales."""
        self._validate_beam(beam)
        beam = beam.float()
        gradient = (spatial_gradient(beam) / self.spatial.gradient_scale).clamp(0, 1)
        logits = self.scale_net(torch.cat((beam, gradient), dim=1))
        weights = torch.softmax(logits, dim=1)
        return (1.0 - self.scale_floor) * weights + self.scale_floor / 3.0

    def forward(self, beam, metadata, namespace, uniform_q=False):
        self._validate_beam(beam)
        if len(metadata) != beam.shape[0]:
            raise ValueError("metadata must contain one entry per beam")
        with torch.autocast(device_type=beam.device.type, enabled=False):
            beam = beam.float()
            weights = self.scale_weights(beam)
            alpha = F.softplus(self.spatial.a)
            beta = F.softplus(self.spatial.d)
            q = probability_map(beam, alpha, beta, self.spatial.gradient_scale)
            r, extent = self.spatial.resolution, self.spatial.extent
            if uniform_q:
                q = torch.full_like(q, 1.0 / (r * r))
            rho = q * (r * r)
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
                    s=(extent, extent),
                    norm="backward",
                ).clamp_min(0.0)
                fields.append(filtered[..., :r, :r])
                variances.append(variance[..., :r, :r])

            guided, _ = combine_scales(
                torch.cat(fields, dim=1), torch.cat(variances, dim=1), weights
            )
            base = self.spatial.samplers["MS"].sample(
                metadata, device=beam.device, namespace=namespace + "-base"
            )
            return math.sqrt(0.75) * base + 0.5 * guided, q, weights
