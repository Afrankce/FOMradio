"""Conditional source translation in normalized dB-map coordinates."""
import torch

KINDS = ('cross_array_observation', 'cross_array_transport')


def source_mean(condition):
    if condition.ndim != 4 or condition.shape[1] != 6:
        raise ValueError('Expected Tx, Height, Beam32, Beam8, Y8, valid channels')
    # Missing source pixels contribute no mean; the unchanged noise is retained.
    return torch.where(condition[:, 5:6] > .5, condition[:, 4:5], 0.)


def initial_state(condition, noise, config):
    if config.get('experiment_kind') != 'cross_array_transport':
        return noise
    if config.get('source_mean_rule') != 'fixed_real_y8_0deg_masked_zero':
        raise ValueError('Unknown conditional source mean')
    if config.get('source_noise_scale') != 1.:
        raise ValueError('This pilot freezes the T2 noise scale at one')
    mean = source_mean(condition)
    if mean.shape != noise.shape:
        raise ValueError('Source observation and noise must share the state grid')
    return mean + noise
