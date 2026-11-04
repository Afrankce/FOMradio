"""Contracts for the isolated finite-anchor Matérn source."""
import importlib.util
import math

import pytest
import torch

torch.set_num_threads(2)


def test_anchor_source_module_exists():
    assert importlib.util.find_spec("anchor_source") is not None, (
        "finite-anchor source is not implemented"
    )


def module():
    return pytest.importorskip("anchor_source")


def beam(size=16):
    y = torch.linspace(0.0, 1.0, size).view(1, 1, size, 1)
    x = torch.linspace(0.0, 1.0, size).view(1, 1, 1, size)
    return (0.65 * x + 0.35 * y).contiguous()


def metadata():
    return [{"scene_id": "synthetic", "steering_deg": 0.0}]


def test_anchor_counts_preserve_budget_and_random_stream():
    source = module()
    q = torch.full((2, 1, 4, 4), 1.0 / 16.0)
    meta = [
        {"scene_id": "a", "steering_deg": 0.0},
        {"scene_id": "b", "steering_deg": 7.0},
    ]
    counts = source.sample_anchor_counts(q, meta, 128, seed=42, namespace="draw")
    repeat = source.sample_anchor_counts(q, meta, 128, seed=42, namespace="draw")
    changed = source.sample_anchor_counts(q, meta, 128, seed=42, namespace="other")
    assert counts.shape == q.shape
    assert bool((counts >= 0).all())
    assert torch.equal(counts.flatten(1).sum(1), torch.tensor([128, 128]))
    assert torch.equal(counts, repeat)
    assert not torch.equal(counts, changed)


def test_anchor_counts_follow_probability_support():
    source = module()
    q = torch.zeros(1, 1, 4, 4)
    q[..., 1, 2] = 1.0
    counts = source.sample_anchor_counts(q, metadata(), 50, seed=42, namespace="draw")
    assert int(counts[..., 1, 2].item()) == 50
    assert int(counts.sum().item()) == 50


def test_invalid_probability_and_count_rejected():
    source = module()
    with pytest.raises(ValueError):
        source.sample_anchor_counts(torch.zeros(1, 1, 4, 4), metadata(), 10, 42, "x")
    with pytest.raises(ValueError):
        source.sample_anchor_counts(torch.ones(1, 1, 4, 4) / 16, metadata(), 0, 42, "x")


def test_zero_sampling_strength_recovers_a_source():
    source = module()
    from model.matern_source import SourceSampler

    b, meta, namespace = beam(), metadata(), "a-match"
    model = source.AnchorMaternSource(
        seed=42, gradient_scale=0.03, resolution=16, anchor_count=64,
        mix_strength=0.0, q_mode="uniform", learn_scales=False,
    )
    noise, q, weights, counts = model(b, meta, namespace)
    samplers = {key: SourceSampler(key, seed=42, resolution=16)
                for key in ("MS", "S2", "S8", "S32")}
    base = samplers["MS"].sample(meta, device=b.device, namespace=namespace + "-base")
    components = sum(samplers[key].sample(meta, device=b.device, namespace=namespace + "-" + key)
                     for key in ("S2", "S8", "S32")) / math.sqrt(3.0)
    expected = math.sqrt(0.75) * base + 0.5 * components
    assert torch.allclose(noise, expected, rtol=2e-5, atol=2e-5)
    assert torch.allclose(q, torch.full_like(q, 1.0 / 256.0))
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0))
    assert int(counts.sum()) == 64


def test_uniform_finite_anchors_change_the_source():
    source = module()
    options = dict(seed=42, gradient_scale=0.03, resolution=16, anchor_count=64,
                   q_mode="uniform", learn_scales=False)
    baseline = source.AnchorMaternSource(**options, mix_strength=0.0)
    finite = source.AnchorMaternSource(**options, mix_strength=0.5)
    z_a, _, _, _ = baseline(beam(), metadata(), "matched")
    z_finite, _, _, _ = finite(beam(), metadata(), "matched")
    assert torch.isfinite(z_finite).all()
    assert float((z_a - z_finite).abs().max()) > 1e-4


def test_learnable_scales_receive_gradient_with_discrete_anchors():
    source = module()
    model = source.AnchorMaternSource(
        seed=42, gradient_scale=0.03, resolution=16, anchor_count=64,
        mix_strength=0.5, q_mode="beam", learn_scales=True,
    )
    z, q, weights, counts = model(beam(), metadata(), "gradient")
    target = torch.linspace(-1.0, 1.0, z.numel()).reshape_as(z)
    (z - target).square().mean().backward()
    gradient = model.scale_net[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0
    assert not model.spatial.a.requires_grad and not model.spatial.d.requires_grad
    assert torch.allclose(q.sum(dim=(-2, -1)), torch.ones(1, 1), atol=1e-6)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1, 16, 16), atol=1e-6)
    assert int(counts.sum()) == 64
