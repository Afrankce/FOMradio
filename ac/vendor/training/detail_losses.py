"""Mask-safe quadratic velocity objectives; L1 is a variant, not an L1 norm."""
import torch
from torch.nn import functional as F
from training.masked_flow_loss import _validate_velocity_tensors


def detail_components(predicted, target, mask):
    _validate_velocity_tensors(predicted, target, mask)
    # Exclude arbitrary invalid values BEFORE any filtering (including NaN).
    error = torch.where(mask, predicted.float() - target.float(), 0.)
    if not torch.isfinite(error).all():
        raise ValueError('Nonfinite valid velocity error')
    terms = {'fm': (error.square().sum(), mask.sum())}
    value = F.avg_pool2d(error, 2)
    valid = F.avg_pool2d(mask.float(), 2) == 1
    numerator = value.sum() * 0
    count = torch.zeros((), device=value.device, dtype=torch.int64)
    for axis in (-2, -1):
        a, b = [slice(None)] * 4, [slice(None)] * 4
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        support = valid[a] & valid[b]
        numerator = numerator + (value[a] - value[b])[support].square().sum()
        count = count + support.sum()
    terms['gradient'] = (numerator, count)
    for level in (1, 2):
        a, b = value[..., ::2, ::2], value[..., ::2, 1::2]
        c, d = value[..., 1::2, ::2], value[..., 1::2, 1::2]
        next_valid = F.avg_pool2d(valid.float(), 2) == 1
        detail = torch.cat(((a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2), 1)
        support = next_valid.expand_as(detail)
        terms[f'haar{level}'] = (detail[support].square().sum(), support.sum())
        value, valid = (a+b+c+d)/2, next_valid
    return terms


def auxiliary(means, variant):
    if variant == 'L1':
        return means['gradient']
    if variant == 'L2':
        return (means['haar1'] + means['haar2']) / 2
    raise ValueError(variant)


def parameter_group(name):
    if '.spectral.' in name:
        return 'spectral'
    if '.local.' in name:
        return 'local'
    if '.attention.' in name:
        return 'attention'
    if 'condition_encoder' in name:
        return 'condition_encoder'
    return 'other'


def gradient_statistics(a, b, weight):
    """Real-coordinate gradients, before clipping/Adam; b is unweighted aux."""
    aa, bb, ab = a.double().square().sum(), b.double().square().sum(), (a.double()*b.double()).sum()
    n, m = float(aa.sqrt()), float(bb.sqrt())
    return dict(fm_norm=n, aux_norm=m, weighted_aux_norm=weight*m,
                weighted_ratio=weight*m/n if n > 0 else None,
                cosine=float(ab/(aa*bb).sqrt()) if n*m > 0 else None,
                total_norm=float((a.double()+weight*b.double()).norm()))
