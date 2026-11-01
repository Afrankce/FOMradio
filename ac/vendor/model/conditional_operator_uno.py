"""Controlled replacements for CA/SA; no source, path, or hidden-noise changes."""
import torch
from torch import nn
from torch.nn import functional as F
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d, AttentionConditionedFNOStage
from model.unet.basic_unet_denose import nonlinearity

VARIANTS = ('CASA', 'ADD', 'FILM', 'NSOP')


class AddCondition(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.embedding_proj = projection

    def forward(self, value, condition):
        return value + self.embedding_proj(condition)


class ConditionModulation(nn.Module):
    """Same conditioner parameter count for FiLM and NSOP at each stage."""
    def __init__(self, channels, width, rank=2):
        super().__init__()
        self.width, self.rank = width, rank
        self.net = nn.Sequential(nn.Conv2d(channels, width, 1), nn.GELU(),
                                 nn.Conv2d(width, 2 * rank * width, 1))
        nn.init.normal_(self.net[-1].weight, std=.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition):
        b, _, h, w = condition.shape
        gates = .5 * torch.tanh(self.net(condition))
        return gates.reshape(b, 2, self.rank, self.width, h, w).unbind(1)


class ConditionalStage(AttentionConditionedFNOStage):
    def forward(self, value, condition, time_embedding):
        self._validate_inputs(value, condition, time_embedding)
        attended = self.attention(value, condition)
        hidden = self.lifting(attended)
        height, width = hidden.shape[-2:]
        a, b = self.condition_operator(condition)
        if self.condition_variant == 'FILM':
            # Generic local affine conditioning; same learned conditioner as NSOP.
            hidden = hidden * (1 + a.mean(1)) + b.mean(1)
        if self._padding:
            hidden = F.pad(hidden, (0, self._padding, 0, self._padding))
        spectral = self.spectral(hidden)
        if self.condition_variant == 'NSOP':
            # K_c = K + (1/J) sum_j M_a_j K M_(1+b_j). R is shared.
            b = 1 + b
            if self._padding:
                a = F.pad(a, (0, self._padding, 0, self._padding))
                b = F.pad(b, (0, self._padding, 0, self._padding))
            for j in range(self.condition_operator.rank):
                spectral = spectral + a[:, j] * self.spectral(b[:, j] * hidden) / self.condition_operator.rank
        temporal = self.time_projection(nonlinearity(time_embedding))[:, :, None, None]
        delta = F.gelu(spectral + self.local(hidden) + temporal)
        return attended + self.projection(delta[..., :height, :width])


class ConditionalOperatorUNO2d(AttentionMultiscaleUNO2d):
    def __init__(self, condition_variant='CASA', **kwargs):
        if condition_variant not in VARIANTS:
            raise ValueError(condition_variant)
        # Construct the identical original model first, preserving all common initial weights.
        super().__init__(**kwargs)
        self.condition_variant = condition_variant
        if condition_variant == 'CASA':
            return
        stages = [*self.encoder_stages, self.bottleneck, *self.decoder_stages]
        # Isolated initializer: substitutions do not advance training/global RNG streams.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(314159)
            for stage in stages:
                stage.attention = AddCondition(stage.attention.embedding_proj)
                if condition_variant in ('FILM', 'NSOP'):
                    stage.__class__ = ConditionalStage
                    stage.condition_variant = condition_variant
                    stage.condition_operator = ConditionModulation(stage._embedding_channels,
                                                                  stage._operator_width)
