"""Isolated AP ablation: reuse the frozen runner, change only condition dropout.

Entrypoint dependency injection is local to this process. Original model files,
optimizer, sampler, loss, synchronization, and evaluation code are unchanged.
"""
from dataclasses import dataclass
import os
import run_spectral_gate as reference


class NoDropUNO(reference.SpectralGateUNO2d):
    def __init__(self, variant='AP', **kwargs):
        if variant != 'AP':
            raise ValueError('This experiment is registered for AP only')
        kwargs['cfg_drop_prob'] = 0.0
        super().__init__(variant, **kwargs)


@dataclass(frozen=True)
class NoDropConfig(reference.SpectralConfig):
    def scientific_payload(self):
        return {**super().scientific_payload(),
                'experiment': 'ap_no_condition_dropout_v1',
                'cfg_drop_prob': 0.0,
                'cfg_dropout_scope': 'disabled; all raw and encoded conditions retained'}


def main():
    if int(os.environ.get('WORLD_SIZE', '1')) != 2:
        raise RuntimeError('Keep the reference two-rank training/evaluation layout')
    reference.SpectralGateUNO2d = NoDropUNO
    reference.SpectralConfig = NoDropConfig
    reference.main()


if __name__ == '__main__':
    main()
