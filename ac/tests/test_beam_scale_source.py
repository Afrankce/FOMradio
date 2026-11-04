"""Numerical contracts for the isolated, three-scale source prototype."""
import importlib.util
import math

import pytest
import torch

torch.set_num_threads(2)


def test_scale_source_module_exists():
    assert importlib.util.find_spec("beam_scale_source") is not None, (
        "Beam-conditioned scale source has not been implemented"
    )


def _module():
    return pytest.importorskip("beam_scale_source")


def _beam(size=16):
    y = torch.linspace(0.0, 1.0, size).view(1, 1, size, 1)
    x = torch.linspace(0.0, 1.0, size).view(1, 1, 1, size)
    return (0.65 * x + 0.35 * y).contiguous()


def _meta():
    return [{"scene_id": "synthetic", "steering_deg": 0.0}]


def test_uniform_scale_weights_match_existing_beam_source():
    module = _module()
    from beam_matern_source import BeamMaternSource

    beam, meta = _beam(), _meta()
    old = BeamMaternSource(seed=42, gradient_scale=0.03, resolution=16)
    new = module.BeamScaleMaternSource(
        seed=42, gradient_scale=0.03, resolution=16
    )
    old_noise, old_q = old(beam, meta, namespace="same-draw")
    new_noise, new_q, weights = new(beam, meta, namespace="same-draw")

    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0))
    assert torch.allclose(new_q, old_q, rtol=1e-6, atol=1e-8)
    assert torch.allclose(new_noise, old_noise, rtol=2e-5, atol=2e-5)


def test_uniform_probability_and_weights_match_a_source():
    module = _module()
    from model.matern_source import SourceSampler

    beam, meta, namespace = _beam(), _meta(), "uniform-draw"
    source = module.BeamScaleMaternSource(seed=42, gradient_scale=0.03, resolution=16)
    actual, q, weights = source(beam, meta, namespace, uniform_q=True)
    samplers = {
        key: SourceSampler(key, seed=42, resolution=16)
        for key in ("MS", "S2", "S8", "S32")
    }
    base = samplers["MS"].sample(meta, device=beam.device, namespace=namespace + "-base")
    equal_scales = sum(
        samplers[key].sample(meta, device=beam.device, namespace=namespace + "-" + key)
        for key in ("S2", "S8", "S32")
    ) / math.sqrt(3.0)
    expected = math.sqrt(0.75) * base + 0.5 * equal_scales
    assert torch.allclose(q, torch.full_like(q, 1.0 / 256.0))
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0))
    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_scale_weights_are_normalized_and_beam_dependent_after_update():
    module = _module()
    source = module.BeamScaleMaternSource(
        seed=42, gradient_scale=0.03, resolution=16
    )
    with torch.no_grad():
        source.scale_net[-1].weight.fill_(0.15)
        source.scale_net[-1].weight[0].mul_(-1.0)
        source.scale_net[-1].weight[1].zero_()
    weights = source.scale_weights(_beam())
    assert weights.shape == (1, 3, 16, 16)
    assert torch.isfinite(weights).all()
    assert bool((weights > 0).all())
    assert torch.allclose(weights.sum(dim=1), torch.ones(1, 16, 16), atol=1e-6)
    assert float(weights.std(dim=(-2, -1)).max()) > 0.0


def test_scale_weights_receive_gradients_through_noise_source():
    module = _module()
    source = module.BeamScaleMaternSource(
        seed=42, gradient_scale=0.03, resolution=16
    )
    noise, _, _ = source(_beam(), _meta(), namespace="gradient-draw")
    target = torch.linspace(-1, 1, noise.numel()).reshape_as(noise)
    (noise - target).square().mean().backward()

    gradient = source.scale_net[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0
    assert not source.spatial.a.requires_grad
    assert not source.spatial.d.requires_grad


def test_variance_normalization_for_nonuniform_scale_weights():
    module = _module()
    generator = torch.Generator().manual_seed(123)
    draws = 4096
    component_variance = torch.tensor([0.25, 1.0, 4.0]).view(1, 3, 1, 1)
    raw = (
        torch.randn(draws, 3, 2, 2, generator=generator)
        * component_variance.sqrt()
    )
    variance = component_variance.expand(draws, 3, 2, 2)
    weights = torch.tensor([0.1, 0.3, 0.6]).view(1, 3, 1, 1)
    weights = weights.expand(draws, 3, 2, 2)
    field, fractions = module.combine_scales(raw, variance, weights)

    empirical = field.var(dim=0, unbiased=True)
    assert torch.allclose(empirical, torch.ones_like(empirical), atol=0.08)
    assert torch.allclose(fractions.sum(dim=1), torch.ones(draws, 2, 2), atol=1e-6)
    expected = weights * variance / (weights * variance).sum(dim=1, keepdim=True)
    assert torch.allclose(fractions, expected, atol=1e-6)


def test_invalid_scale_floor_is_rejected():
    module = _module()
    with pytest.raises(ValueError):
        module.BeamScaleMaternSource(
            seed=42, gradient_scale=0.03, resolution=16, scale_floor=1.1
        )
