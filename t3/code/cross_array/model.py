import torch
from torch import nn
from model.spectral_gate_uno import SpectralGateUNO2d


class SourceObservationUNO(SpectralGateUNO2d):
    """Matched six-channel backbone; T0 hides source values and their mask."""
    def __init__(self, arm):
        if arm not in ('T0', 'T2', 'T3'): raise ValueError(arm)
        super().__init__('AP', operator_width=32, cfg_drop_prob=0.)
        self.arm = arm; self.condition_channels = 6
        # Expand only input layers; keep all old common tensors and initial
        # predictions identical. No additional nonstationary residual branch.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(161803)
            name, old = next((n, m) for n, m in self.condition_encoder.named_modules() if isinstance(m, nn.Conv2d))
            assert old.in_channels == 3
            new = nn.Conv2d(6, old.out_channels, old.kernel_size, stride=old.stride,
                padding=old.padding, dilation=old.dilation, groups=old.groups, bias=old.bias is not None)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :3].copy_(old.weight)
                if old.bias is not None: new.bias.copy_(old.bias)
            parent, _, leaf = name.rpartition('.')
            setattr(self.condition_encoder.get_submodule(parent), leaf, new)
            old = self.lifting; new = nn.Conv2d(9, old.out_channels, 1)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :4].copy_(old.weight[:, :4])
                new.weight[:, 7:9].copy_(old.weight[:, 4:6]); new.bias.copy_(old.bias)
            self.lifting = new

    def sanitize(self, condition):
        self._validate_condition(condition)
        if self.arm == 'T0': return torch.cat((condition[:, :4], torch.zeros_like(condition[:, 4:6])), 1)
        return condition

    def embed_model(self, condition):
        return super().embed_model(self.sanitize(condition))

    def forward(self, image=None, x=None, pred_type='denoise', step=None, embedding=None):
        return super().forward(image=self.sanitize(image), x=x, pred_type=pred_type, step=step, embedding=embedding)
