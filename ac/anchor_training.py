"""Common model/optimizer contracts for the finite-anchor A variants."""
import hashlib

import torch
from torch import nn

from anchor_source import AnchorMaternSource
from cross_array.model import SourceObservationUNO


def seed_for(*values):
    text = "|".join(map(str, values)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") % (2**63 - 1)


class AnchorSystem(nn.Module):
    def __init__(self, seed, gradient_scale, q_mode, learn_scales, resolution=256,
                 anchor_count=4096, mix_strength=0.5, scale_floor=0.05):
        super().__init__()
        torch.manual_seed(seed)
        self.velocity = SourceObservationUNO("T0")
        # Keep the velocity's exact A initialization and give each variant the
        # same independently seeded scale-network initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_for(seed, "anchor-scale-network"))
            self.source = AnchorMaternSource(
                seed=seed, gradient_scale=gradient_scale, resolution=resolution,
                anchor_count=anchor_count, mix_strength=mix_strength,
                q_mode=q_mode, learn_scales=learn_scales, scale_floor=scale_floor,
            )


def build_system(seed, gradient_scale, q_mode, learn_scales, **kwargs):
    return AnchorSystem(seed, gradient_scale, q_mode, learn_scales, **kwargs)


def make_optimizer(system):
    return torch.optim.AdamW(
        [p for p in system.parameters() if p.requires_grad],
        lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5, foreach=False,
    )
